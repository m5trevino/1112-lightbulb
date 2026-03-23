// netlify/functions/visitor_counter.js
export const handler = async () => {
  const time = new Date().getTime();
  const count = Math.floor(time / 1000) % 10000; // A number that changes every second

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ count }),
  };
};
