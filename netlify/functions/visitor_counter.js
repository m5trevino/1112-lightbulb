// netlify/functions/visitor_counter.js
import { NetlifyKV } from "@netlify/kv";

// Define a specific store for this site
const STORE_NAME = "1112_king_rd_forensics";

export const handler = async (event) => {
  try {
    const store = new NetlifyKV(STORE_NAME);
    let count = await store.get("visitor_count");
    
    // If running locally or count is null, start at 1
    if (count === null) {
      count = 0;
    }

    const newCount = count + 1;
    await store.set("visitor_count", newCount);

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count: newCount }),
    };
  } catch (error) {
    console.error("Error in visitor_counter function:", error);
    
    // Fallback for local development if KV store isn't available
    if (process.env.NETLIFY_DEV) {
        const dummyCount = Math.floor(Math.random() * (9999 - 1000 + 1) + 1000);
        return {
            statusCode: 200,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ count: dummyCount, note: "Running in dev mode" }),
        };
    }

    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Could not process visitor count." }),
    };
  }
};
