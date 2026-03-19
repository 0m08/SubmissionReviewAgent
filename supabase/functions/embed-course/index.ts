import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const COHERE_API_KEY = Deno.env.get("COHERE_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

serve(async (req) => {
  try {
    const payload = await req.json();
    const record = payload.record;

    if (!record) {
      return new Response(JSON.stringify({ error: "No record in payload" }), {
        status: 400,
      });
    }

    // Skip if embedding already exists (avoid infinite loop from our own UPDATE)
    if (record.embedding) {
      return new Response(JSON.stringify({ skipped: true, reason: "embedding already exists" }), {
        status: 200,
      });
    }

    const pageContent = record.page_content;
    if (!pageContent || !pageContent.trim()) {
      return new Response(JSON.stringify({ skipped: true, reason: "no page_content" }), {
        status: 200,
      });
    }

    // Call Cohere embed API
    const cohereResp = await fetch("https://api.cohere.ai/v1/embed", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${COHERE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts: [pageContent],
        model: "embed-english-v3.0",
        input_type: "search_document",
        truncate: "END",
      }),
    });

    if (!cohereResp.ok) {
      const errText = await cohereResp.text();
      console.error("Cohere API error:", errText);
      return new Response(JSON.stringify({ error: "Cohere API failed", details: errText }), {
        status: 500,
      });
    }

    const cohereData = await cohereResp.json();
    const embedding = cohereData.embeddings[0];

    // Write embedding back to the row
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
    const { error } = await supabase
      .from("courses")
      .update({ embedding: JSON.stringify(embedding) })
      .eq("id", record.id);

    if (error) {
      console.error("Supabase update error:", error.message);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
      });
    }

    console.log(`Embedded course id=${record.id} doc_id=skillcat_${record.course_id}`);
    return new Response(
      JSON.stringify({ success: true, course_id: record.course_id }),
      { status: 200 }
    );
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
    });
  }
});
