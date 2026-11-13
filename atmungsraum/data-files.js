// ZIP32 with stored entries preserves the original recording bytes.
const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  return value >>> 0;
});
export function createExportsZip(records) {
  const encoder = new TextEncoder(), parts = [], directory = [], names = new Set();
  let offset = 0, directorySize = 0;
  if (records.length > 65535) throw new Error('Too many ZIP entries');
  for (const record of records) {
    const path = names.has(record.filename) ? `${record.session_id}/${record.filename}` : record.filename;
    if (names.has(path) || /[\\/\0]/.test(record.filename) || /[\\/\0]/.test(record.session_id)) throw new Error('Invalid ZIP filename');
    names.add(path);
    const name = encoder.encode(path), bytes = record.bytes ?? encoder.encode(record.csv);
    if (name.length > 65535 || offset + bytes.length + name.length + 30 > 0xffffffff) throw new Error('ZIP32 size exceeded');
    let crc = 0xffffffff;
    for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 255];
    crc = (crc ^ 0xffffffff) >>> 0;
    const local = new Uint8Array(30), header = new DataView(local.buffer);
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true);
    header.setUint16(6, 0x0800, true); // UTF-8 filenames.
    header.setUint16(12, 33, true); // 1980-01-01; recording UTC stays in the CSV.
    header.setUint32(14, crc, true);
    header.setUint32(18, bytes.length, true);
    header.setUint32(22, bytes.length, true);
    header.setUint16(26, name.length, true);
    const central = new Uint8Array(46), entry = new DataView(central.buffer);
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    central.set(local.subarray(4, 30), 6);
    entry.setUint32(42, offset, true);
    parts.push(local, name, bytes);
    directory.push(central, name);
    directorySize += central.length + name.length;
    offset += local.length + name.length + bytes.length;
  }
  if (offset + directorySize + 22 > 0xffffffff) throw new Error('ZIP32 size exceeded');
  const end = new Uint8Array(22), footer = new DataView(end.buffer);
  footer.setUint32(0, 0x06054b50, true);
  footer.setUint16(8, records.length, true);
  footer.setUint16(10, records.length, true);
  footer.setUint32(12, directorySize, true);
  footer.setUint32(16, offset, true);
  return new Blob([...parts, ...directory, end], { type: 'application/zip' });
}
