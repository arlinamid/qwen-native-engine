// Tool declarations (the shape the chat template expects) and demo implementations.
const fn = (name, description, properties = {}, required = []) => ({
  type: 'function',
  function: { name, description, parameters: { type: 'object', properties, required } },
});

export const TOOLS = [
  fn('get_weather', 'Get the current weather for a city.', {
    city: { type: 'string', description: 'City name' },
    unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
  }, ['city']),
  fn('calculate', 'Evaluate an arithmetic expression.', {
    expression: { type: 'string', description: 'For example 12*(3+4)' },
  }, ['expression']),
  fn('set_timer', 'Start a countdown timer.', {
    minutes: { type: 'number' },
    label: { type: 'string' },
  }, ['minutes']),
  fn('search_notes', "Search the user's personal notes.", {
    query: { type: 'string' },
  }, ['query']),
  fn('send_message', 'Send a text message to a person.', {
    recipient: { type: 'string' },
    text: { type: 'string' },
  }, ['recipient', 'text']),
  fn('get_datetime', 'Get the current date and time.', {
    timezone: { type: 'string', description: 'IANA time zone, for example Europe/Budapest' },
  }),
];

export const SYSTEM_PROMPT = 'You are a helpful assistant. Use a tool only when it is needed to fulfil the request. Answer in the language of the user.';

// A small recursive-descent parser, so the model's expression never reaches eval().
function evaluate(source) {
  let i = 0;
  const peek = () => source[i];
  const skip = () => { while (peek() === ' ') i++; };
  const number = () => {
    skip();
    if (peek() === '(') {
      i++;
      const value = sum();
      skip();
      if (peek() !== ')') throw new Error('missing )');
      i++;
      return value;
    }
    if (peek() === '-') { i++; return -number(); }
    const start = i;
    while (/[0-9.]/.test(peek() ?? '')) i++;
    if (start === i) throw new Error(`unexpected "${peek() ?? 'end'}"`);
    return Number(source.slice(start, i));
  };
  const product = () => {
    let value = number();
    for (skip(); peek() === '*' || peek() === '/'; skip()) {
      const op = source[i++];
      value = op === '*' ? value * number() : value / number();
    }
    return value;
  };
  function sum() {
    let value = product();
    for (skip(); peek() === '+' || peek() === '-'; skip()) {
      const op = source[i++];
      value = op === '+' ? value + product() : value - product();
    }
    return value;
  }
  const value = sum();
  skip();
  if (i !== source.length) throw new Error(`unexpected "${peek()}"`);
  return value;
}

export function runTool(name, args) {
  try {
    switch (name) {
      case 'calculate':
        return JSON.stringify({ result: evaluate(args.expression) });
      case 'get_datetime':
        return JSON.stringify({
          now: new Date().toLocaleString('sv-SE', args.timezone ? { timeZone: args.timezone } : {}),
          timezone: args.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
        });
      case 'get_weather':
        return JSON.stringify({ city: args.city, temp_c: 18, condition: 'cloudy', note: 'demo data' });
      case 'set_timer':
        return JSON.stringify({ ok: true, minutes: args.minutes, label: args.label ?? null, note: 'demo: no timer started' });
      case 'search_notes':
        return JSON.stringify({ results: [], note: 'demo: no notes connected' });
      case 'send_message':
        return JSON.stringify({ ok: false, note: 'demo: nothing was sent' });
      default:
        return JSON.stringify({ error: `unknown tool ${name}` });
    }
  } catch (error) {
    return JSON.stringify({ error: String(error.message) });
  }
}
