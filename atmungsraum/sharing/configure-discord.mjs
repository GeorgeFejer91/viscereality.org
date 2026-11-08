// Run locally. Only discord-config.json may be committed; secrets stay ignored.
import { randomBytes } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { encryptDiscordWebhook, decryptDiscordWebhook } from '../discord-share.js';

const endpoint = process.argv[2];
if (!endpoint) throw new Error('Pass the deployed upload service URL to configure-discord.mjs.');
const privateVars = await readFile(new URL('.dev.vars', import.meta.url), 'utf8');
const webhook = privateVars.match(/^DISCORD_WEBHOOK_URL=(?:"([^"\r\n]+)"|([^\r\n]+))$/m)?.slice(1).find(Boolean)?.trim();
if (!webhook) throw new Error('Set DISCORD_WEBHOOK_URL in the ignored sharing/.dev.vars first.');
const passwordFile = new URL('.dev.vars.discord-password', import.meta.url);
try { await writeFile(passwordFile, randomBytes(18).toString('base64url') + '\n', { flag: 'wx' }); }
catch (error) { if (error.code !== 'EEXIST') throw error; }
const password = (await readFile(passwordFile, 'utf8')).replace(/^\uFEFF/, '').replace(/\r?\n$/, '');
if (/[\r\n]/.test(password)) throw new Error('Use a single-line password.');
const token = (await readFile(new URL('.dev.vars.render-token', import.meta.url), 'utf8')).trim();
const config = await encryptDiscordWebhook(webhook, password, { endpoint, token });
if (await decryptDiscordWebhook(config, password) !== webhook) throw new Error('Round-trip verification failed.');
await writeFile(new URL('discord-config.json', import.meta.url), JSON.stringify(config, null, 2) + '\n');
await writeFile(new URL('.dev.vars.setup.html', import.meta.url), `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Private Discord sharing password</title><style>body{max-width:700px;margin:40px auto;padding:20px;font:18px/1.6 system-ui}input{font:inherit;width:100%;box-sizing:border-box;padding:10px}button{font:inherit;padding:10px;margin:8px 0}</style><h1>Private Discord sharing password</h1><p>Keep this file private. The public website contains only the encrypted delivery credentials.</p><p>Password (select to copy):</p><input readonly type="password" autocomplete="off" value="${password.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;')}" aria-label="Discord sharing password" onclick="this.select()"><button onclick="const i=document.querySelector('input');i.type=i.type==='password'?'text':'password'">Show/hide password</button><ol><li>Open <a href="https://viscereality.org/atmungsraum/" rel="noreferrer">Atmungsraum</a> and unlock delivery with this password. You can let a supported browser remember it.</li><li>Open the bottom-right folder and choose Ungesendete Daten senden or Alle Daten senden. The app uploads the encrypted ZIP and posts only the Wormhole URL to Discord.</li></ol><p>Anyone who knows this password can recover the webhook and the upload key; share it only with intended operators.</p><p>To change the password, edit the ignored .dev.vars.discord-password file (one nonempty line), run node sharing/configure-discord.mjs again, and publish the new encrypted JSON. Rotate the webhook in Discord and the relay key on the server to revoke existing copies.</p>`);
console.log('Encrypted sharing/discord-config.json created and verified.');
console.log('Password stays in ignored sharing/.dev.vars.discord-password and .dev.vars.setup.html; no secrets printed.');
