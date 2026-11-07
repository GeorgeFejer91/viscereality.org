// Run locally. Only discord-config.json may be committed; secrets stay ignored.
import { randomBytes } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { encryptDiscordWebhook, decryptDiscordWebhook } from '../discord-share.js';

const privateVars = await readFile(new URL('.dev.vars', import.meta.url), 'utf8');
const webhook = privateVars.match(/^DISCORD_WEBHOOK_URL=(?:"([^"\r\n]+)"|([^\r\n]+))$/m)?.slice(1).find(Boolean)?.trim();
if (!webhook) throw new Error('Set DISCORD_WEBHOOK_URL in the ignored sharing/.dev.vars first.');
const passwordFile = new URL('.dev.vars.discord-password', import.meta.url);
try { await writeFile(passwordFile, randomBytes(18).toString('base64url') + '\n', { flag: 'wx' }); }
catch (error) { if (error.code !== 'EEXIST') throw error; }
const password = (await readFile(passwordFile, 'utf8')).trim();
if (/[\r\n]/.test(password)) throw new Error('Use a single-line password.');
const config = await encryptDiscordWebhook(webhook, password);
if (await decryptDiscordWebhook(config, password) !== webhook) throw new Error('Round-trip verification failed.');
await writeFile(new URL('discord-config.json', import.meta.url), JSON.stringify(config, null, 2) + '\n');
await writeFile(new URL('.dev.vars.setup.html', import.meta.url), `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Private Discord sharing password</title><style>body{max-width:700px;margin:40px auto;padding:20px;font:18px/1.6 system-ui}input{font:inherit;width:100%;box-sizing:border-box;padding:10px}button{font:inherit;padding:10px;margin:8px 0}</style><h1>Private Discord sharing password</h1><p>Keep this file private. The public website contains only the encrypted webhook.</p><p>Password (select to copy):</p><input readonly type="password" autocomplete="off" value="${password.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;')}" aria-label="Discord sharing password" onclick="this.select()"><button onclick="const i=document.querySelector('input');i.type=i.type==='password'?'text':'password'">Show/hide password</button><ol><li>Open <a href="https://viscereality.org/atmungsraum/" rel="noreferrer">Atmungsraum</a> → folder → Per QR-Code teilen.</li><li>Upload the ZIP on Wormhole and paste its complete link, or use an already configured automatic upload.</li><li>Choose An Discord senden (or leave automatic forwarding checked), enter this password when prompted and press Senden. Cancelling sends nothing.</li></ol><p>No Cloudflare account is needed for Discord delivery. Anyone who knows this password can recover the webhook; share it only with intended operators.</p><p>To change the password, edit the ignored .dev.vars.discord-password file (at least 16 characters), run node sharing/configure-discord.mjs again, and publish the new encrypted JSON. Existing decrypted copies of the webhook remain valid until it is rotated in Discord.</p>`);
console.log('Encrypted sharing/discord-config.json created and verified.');
console.log('Password stays in ignored sharing/.dev.vars.discord-password and .dev.vars.setup.html; no secrets printed.');
