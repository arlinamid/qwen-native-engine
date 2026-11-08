import test from 'node:test';
import assert from 'node:assert/strict';
import {scoreCase, summarize} from '../web/harness.js';
const c = {expected:[{name:'calculate',args:{expression:'1234*56'}}],answer_contains:['69104']};
const good = {calls:[{name:'calculate',arguments:{expression:'1234 * 56'}}],parse_errors:0,finished:true,answer:'69104'};
test('routing alone, incomplete output, and incorrect answers do not pass', () => {
  assert.equal(scoreCase(c,good).pass,true);
  for (const changed of [{finished:false},{answer:''},{answer:'1234'}, {parse_errors:1}, {calls:[]}, {calls:[{...good.calls[0],error:'failed'}]}]) {
    assert.equal(scoreCase(c,{...good,...changed}).pass,false);
  }
  const pending=scoreCase({expected:c.expected},good);
  assert.equal(pending.pass,null);
  assert.equal(summarize([{...pending,parseErrors:0,invalidCalls:0,tokens:0,ms:0}]).all.pending_review,1);
});
