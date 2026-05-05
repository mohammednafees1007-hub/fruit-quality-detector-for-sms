import { NextResponse } from "next/server";
import { normalizeBackendResult } from "@/lib/backendAdapter";
import { runMockDetection } from "@/lib/mockDetection";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const image = formData.get("image");

    if (!(image instanceof File)) {
      return NextResponse.json({ error: "Upload a fruit image first." }, { status: 400 });
    }

    if (process.env.DETECTION_MODE === "backend") {
      const backendUrl = process.env.FASTAPI_DETECT_URL || "http://127.0.0.1:8000/detect";
      const backendForm = new FormData();
      backendForm.append("file", image, image.name || "fruit-scan.jpg");

      const robustCamera = formData.get("robust_camera");
      if (robustCamera) backendForm.append("robust_camera", String(robustCamera));

      const response = await fetch(backendUrl, {
        method: "POST",
        body: backendForm
      });

      const payload = await response.json();
      if (!response.ok) {
        return NextResponse.json(
          { error: payload?.detail || "FastAPI backend could not process this fruit image." },
          { status: response.status }
        );
      }

      return NextResponse.json(normalizeBackendResult(payload));
    }

    return NextResponse.json(await runMockDetection(image.name));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Detection failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
