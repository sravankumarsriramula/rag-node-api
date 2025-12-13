export function chunkText(text, maxLength = 2000) {
  const sentences = text.split(/(?<=[.?!])\s+/);
  const chunks = [];
  let current = "";

  for (const sentence of sentences) {
    if (!sentence.trim()) continue;

    if ((current + " " + sentence).length > maxLength) {
      if (current.trim().length) chunks.push(current.trim());
      current = sentence;
    } else {
      current += " " + sentence;
    }
  }

  if (current.trim().length) chunks.push(current.trim());

  return chunks;
}
 