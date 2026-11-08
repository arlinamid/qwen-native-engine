//! Explicit, bounded JSON Schema subset. Unsupported constraints are rejected, never ignored.
use serde_json::Value;
pub fn check(schema: &Value) -> Result<(), String> { check_at(schema,0) }
fn check_at(schema: &Value, depth: usize) -> Result<(),String> {
    if depth>24 { return Err("schema nesting exceeds 24".into()); }
    if schema.is_boolean() { return Ok(()); }
    let map=schema.as_object().ok_or("schema must be an object or boolean")?;
    for (key,value) in map {
        match key.as_str() {
            "$schema"|"$id"|"title"|"description"|"default"|"examples"|"deprecated"|"readOnly"|"writeOnly" => {},
            "type" => {
                let kind=value.as_str().ok_or("union types are not supported by the tool grammar")?;
                if !matches!(kind,"string"|"number"|"integer"|"boolean"|"object"|"array"|"null") { return Err("unsupported schema type".into()); }
            }
            "properties" => for child in value.as_object().ok_or("properties must be an object")?.values() { check_at(child,depth+1)?; },
            "items"|"additionalProperties" => check_at(value,depth+1)?,
            "required" => {
                for key in value.as_array().ok_or("required must be an array")? {
                    let key=key.as_str().ok_or("invalid required name")?;
                    if schema.get("properties").and_then(|p|p.get(key)).is_none() { return Err(format!("required property {key} has no schema")); }
                }
            }
            "enum" => if value.as_array().is_none_or(|a|a.is_empty()) { return Err("enum must be non-empty".into()); },
            "const" => {},
            "minimum"|"maximum"|"exclusiveMinimum"|"exclusiveMaximum" => if !value.is_number() { return Err(format!("{key} must be numeric")); },
            "minLength"|"maxLength"|"minItems"|"maxItems"|"minProperties"|"maxProperties" => if value.as_u64().is_none() { return Err(format!("{key} must be a nonnegative integer")); },
            "uniqueItems" => if !value.is_boolean() { return Err("uniqueItems must be boolean".into()); },
            _ => return Err(format!("unsupported JSON Schema keyword: {key}")),
        }
    }
    Ok(())
}
fn matches_type(kind:&str, value:&Value) -> bool {
    match kind {
        "string"=>value.is_string(),"number"=>value.is_number(),
        "integer"=>value.is_number() && value.as_f64().is_some_and(|n| n.fract()==0.0),
        "boolean"=>value.is_boolean(),"object"=>value.is_object(),"array"=>value.is_array(),"null"=>value.is_null(),_=>false,
    }
}
pub fn validate(schema:&Value, value:&Value) -> Result<(),String> {
    check(schema)?;
    validate_at(schema,value,"arguments")
}
fn validate_at(schema:&Value,value:&Value,path:&str) -> Result<(),String> {
    if schema==&Value::Bool(true) {return Ok(());}
    if schema==&Value::Bool(false) {return Err(format!("{path} is forbidden"));}
    if let Some(types)=schema.get("type") {
        let valid=if let Some(t)=types.as_str() { matches_type(t,value) } else { types.as_array().unwrap().iter().any(|t|matches_type(t.as_str().unwrap(),value)) };
        if !valid {return Err(format!("{path} must be of type {}",types.as_str().unwrap_or("declared union")));}
    }
    if let Some(options)=schema.get("enum").and_then(Value::as_array) {if !options.contains(value) {return Err(format!("{path} must be one of {options:?}"));}}
    if let Some(expected)=schema.get("const") {if expected!=value {return Err(format!("{path} does not match const"));}}
    let bound=|key:&str|schema.get(key).and_then(Value::as_u64);
    let length=|n:usize,min:&str,max:&str| -> Result<(),String> {
        if bound(min).is_some_and(|b|(n as u64)<b) || bound(max).is_some_and(|b|(n as u64)>b) {Err(format!("{path} violates {min}/{max}"))} else {Ok(())}
    };
    if let Some(n)=value.as_f64() {
        for (key,invalid) in [("minimum",false),("maximum",false),("exclusiveMinimum",true),("exclusiveMaximum",true)] {
            if let Some(b)=schema.get(key).and_then(Value::as_f64) {
                let lower=key.ends_with("Minimum") || key=="minimum";
                if if lower { n<b || (invalid&&n==b) } else { n>b || (invalid&&n==b) } {return Err(format!("{path} violates {key}"));}
            }
        }
    }
    if let Some(s)=value.as_str() {length(s.chars().count(),"minLength","maxLength")?;}
    if let Some(items)=value.as_array() {
        length(items.len(),"minItems","maxItems")?;
        for (i,item) in items.iter().enumerate() {
            if schema["uniqueItems"]==true && items[..i].contains(item) {return Err(format!("{path} items must be unique"));}
            if let Some(child)=schema.get("items") {validate_at(child,item,&format!("{path}[{i}]"))?;}
        }
    }
    if let Some(props)=value.as_object() {
        length(props.len(),"minProperties","maxProperties")?;
        if let Some(required)=schema.get("required").and_then(Value::as_array) {
            for key in required {let key=key.as_str().unwrap(); if !props.contains_key(key) {return Err(format!("missing required argument '{key}' at {path}"));}}
        }
        for (key,item) in props {
            if let Some(child)=schema.get("properties").and_then(|p|p.get(key)) {validate_at(child,item,&format!("{path}.{key}"))?;}
            else if let Some(additional)=schema.get("additionalProperties") {
                if additional==false {return Err(format!("unknown argument '{key}' at {path}"));}
                validate_at(additional,item,&format!("{path}.{key}"))?;
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*; use serde_json::json;
    #[test] fn nested_bounds_and_additional_properties() {
        let schema=json!({"type":"object","properties":{"rows":{"type":"array","minItems":1,"items":{"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":5}},"required":["n"],"additionalProperties":false}}},"required":["rows"]});
        assert!(validate(&schema,&json!({"rows":[{"n":3}],"allowed_extra":1})).is_ok());
        for bad in [json!({"rows":[]}),json!({"rows":[{"n":0}]}),json!({"rows":[{"n":2,"extra":0}]}),json!({"rows":[{}]})] {assert!(validate(&schema,&bad).is_err());}
        assert!(check(&json!({"pattern":".*"})).is_err());
        assert!(check(&json!({"$ref":"https://example.com/schema"})).is_err());
    }
}
