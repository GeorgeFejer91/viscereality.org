import test from 'node:test';
import assert from 'node:assert/strict';

test('one password unlocks the Discord webhook and fixed upload relay without exposing either secret', async t => {
  const { encryptDiscordWebhook, unlockSharingConfig } = await import('../discord-share.js');
  const webhook = 'https://discord.com/api/webhooks/123456789/TEST_TOKEN';
  const password = 'synthetic-long-password-42';
  const relay = { endpoint: 'https://atmungsraum-test.onrender.com', token: 'A'.repeat(43) };
  const config = await encryptDiscordWebhook(webhook, password, relay);
  for (const value of [webhook, password, relay.token]) assert.ok(!JSON.stringify(config).includes(value));
  const calls = [];
  let envelope = config;
  t.mock.method(globalThis, 'fetch', async target => {
    calls.push(String(target));
    return Response.json(envelope);
  });
  assert.deepEqual(await unlockSharingConfig(password), { webhook, relay });
  await assert.rejects(() => unlockSharingConfig('wrong-password'), /Passwort/);
  envelope = await encryptDiscordWebhook(webhook, password);
  assert.deepEqual(await unlockSharingConfig(password), { webhook });
  assert.equal(calls.length, 3);
  assert.ok(calls.every(target => target.endsWith('/sharing/discord-config.json')));
  await assert.rejects(() => encryptDiscordWebhook(webhook, password, { ...relay, endpoint: 'https://attacker.test' }));
});

test('A password encrypts the webhook into public ciphertext and only the correct password unlocks it', async () => {
  const { encryptDiscordWebhook, decryptDiscordWebhook } = await import('../discord-share.js');
  const webhook='https://discord.com/api/webhooks/123456789/TEST_TOKEN', password='synthetic-long-password-42';
  const config=await encryptDiscordWebhook(webhook,password), second=await encryptDiscordWebhook(webhook,password);
  assert.equal(config.version,1); assert.equal(config.iterations,600000);
  assert.ok(!JSON.stringify(config).includes(webhook)); assert.ok(!JSON.stringify(config).includes('TEST_TOKEN')); assert.ok(!JSON.stringify(config).includes(password));
  assert.notEqual(config.salt,second.salt); assert.notEqual(config.iv,second.iv); assert.notEqual(config.ciphertext,second.ciphertext);
  assert.equal(await decryptDiscordWebhook(config,password),webhook);
  await assert.rejects(()=>decryptDiscordWebhook(config,'incorrect-password'),/Passwort/);
  await assert.rejects(()=>decryptDiscordWebhook({...config,ciphertext:(config.ciphertext[0]==='A'?'B':'A')+config.ciphertext.slice(1)},password),/Passwort/);
  for(const changed of [{iterations:1},{iterations:1e12},{salt:'bad'},{iv:'bad'},{ciphertext:'A'.repeat(10000)},{version:2}]) await assert.rejects(()=>decryptDiscordWebhook({...config,...changed},password));
  await assert.rejects(()=>encryptDiscordWebhook(webhook,'short'));
});

test('Password forwarding loads only local ciphertext, unlocks on-device, and sends straight to Discord', async t => {
  const { encryptDiscordWebhook, forwardPasswordDiscordLink } = await import('../discord-share.js');
  const webhook='https://discord.com/api/webhooks/123456789/TEST_TOKEN',password='synthetic-long-password-42',url='https://wormhole.app/room01#fnQhVRrakH6ZUgQyUeAFfA';
  const config=await encryptDiscordWebhook(webhook,password),calls=[];
  t.mock.method(globalThis,'fetch',async(target,init)=>{calls.push({target:String(target),init});return calls.length===1?Response.json(config):Response.json({id:'accepted'})});
  const result=await forwardPasswordDiscordLink({password,url});
  assert.equal(result.destination,'discord'); assert.equal(calls.length,2); assert.match(calls[0].target,/\/sharing\/discord-config.json$/);
  assert.equal(calls[0].init.body,undefined);assert.ok(!calls[0].target.includes(password));
  assert.equal(calls[1].target,webhook+'?wait=true'); assert.ok(!calls[1].init.body.includes(password));
  const controller=new AbortController();controller.abort();await assert.rejects(()=>forwardPasswordDiscordLink({password,url,signal:controller.signal}));assert.equal(calls.length,2);
  t.mock.restoreAll();let attempts=0;t.mock.method(globalThis,'fetch',async()=>{attempts++;return Response.json(config)});
  await assert.rejects(()=>forwardPasswordDiscordLink({password:'wrong-password',url}));assert.equal(attempts,1,'wrong password must never contact Discord');
});
