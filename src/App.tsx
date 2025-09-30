import React, { useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

type LM = { x: number; y: number; z?: number };

const METHODOLOGY = {
  weights: { symmetry: 0.35, golden: 0.25, harmony: 0.40 },
  eyeSpacing: { idealMin: 0.28, idealMax: 0.36, low: 0.26, high: 0.38 },
  quality: { blurVarMin: 1500, luminanceMin: 60, luminanceMax: 200 },
  uncertainty: { repeats: 40, jitterSigma: 0.003 },
  version: "1.2.0",
};

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-black/[0.06] bg-white/70 shadow-[0_10px_35px_-15px_rgba(0,0,0,0.25)] backdrop-blur-sm ${className}`}>
      {children}
    </div>
  );
}
function CardBody({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`p-6 md:p-8 ${className}`}>{children}</div>;
}
function Button({
  children,
  onClick,
  disabled = false,
  variant = "primary",
  size = "md",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "outline";
  size?: "sm" | "md" | "lg";
}) {
  const base =
    "inline-flex items-center justify-center rounded-xl font-medium transition-colors focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed";
  const sizes: Record<string, string> = {
    sm: "px-3 py-2 text-sm",
    md: "px-4 py-2.5 text-sm",
    lg: "px-5 py-3 text-base",
  };
  const variants: Record<string, string> = {
    primary: "bg-neutral-900 text-white hover:bg-black",
    secondary: "bg-white text-neutral-900 border border-neutral-300 hover:bg-neutral-100",
    outline: "bg-transparent text-neutral-900 border border-neutral-300 hover:bg-neutral-50",
  };
  return (
    <button onClick={onClick} disabled={disabled} className={`${base} ${sizes[size]} ${variants[variant]}`}>
      {children}
    </button>
  );
}
function Progress({ value, className = "" }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={`w-full h-2 rounded-full bg-neutral-200 ${className}`}>
      <div className="h-full rounded-full bg-neutral-900 transition-[width] duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const round1 = (v: number) => Math.round(v * 10) / 10;
const round2 = (v: number) => Math.round(v * 100) / 100;
const round3 = (v: number) => Math.round(v * 1000) / 1000;
const euclid = (p1: [number, number], p2: [number, number]) => Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);

const symmetricPairs: Array<[number, number]> = [
  [234, 454], [93, 323], [132, 361], [58, 288], [127, 356],
  [50, 280], [101, 330], [205, 425], [98, 327], [55, 285],
  [65, 295], [107, 336], [52, 282], [66, 296], [3, 13],
];

function calcSymmetryScore(landmarks: LM[]): number {
  const diffs = symmetricPairs.map(([i, j]) => Math.abs(landmarks[i].x - (1 - landmarks[j].x)));
  const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  return Math.max(0, Math.round((100 - mean * 1000) * 100) / 100);
}

function project(idx: number, w: number, h: number, lms: LM[]): [number, number] {
  return [lms[idx].x * w, lms[idx].y * h];
}

type GoldenOut = {
  score: number;
  values: Record<string, number | Record<string, number>>;
  faceWidth: number;
  eyeDist: number;
  eyeL: [number, number];
  eyeR: [number, number];
  ratios: Record<string, number>;
  targets: Record<string, number>;
};

function analyzeGolden(imageW: number, imageH: number, lms: LM[]): GoldenOut {
  const top = project(10, imageW, imageH, lms);
  const bottom = project(152, imageW, imageH, lms);
  const left = project(234, imageW, imageH, lms);
  const right = project(454, imageW, imageH, lms);
  const eyeL = project(133, imageW, imageH, lms);
  const eyeR = project(362, imageW, imageH, lms);
  const mouthL = project(61, imageW, imageH, lms);
  const mouthR = project(291, imageW, imageH, lms);
  const nose = project(1, imageW, imageH, lms);
  const chin = project(152, imageW, imageH, lms);
  const lipU = project(13, imageW, imageH, lms);
  const lipD = project(14, imageW, imageH, lms);

  const faceLength = euclid(top, bottom);
  const faceWidth = euclid(left, right);
  const eyeDist = euclid(eyeL, eyeR);
  const mouthWidth = euclid(mouthL, mouthR);
  const noseToChin = euclid(nose, chin);
  const lipHeight = euclid(lipU, lipD);

  const ratios: Record<string, number> = {
    "Rapport Longueur/ Largeur du visage": faceWidth > 0 ? faceLength / faceWidth : 0,
    "Distance inter-oculaire / Largeur bouche": mouthWidth > 0 ? eyeDist / mouthWidth : 0,
    "Distance inter-oculaire / Largeur visage": faceWidth > 0 ? eyeDist / faceWidth : 0,
    "Nez→Menton / Longueur visage": faceLength > 0 ? noseToChin / faceLength : 0,
    "Hauteur lèvres / Largeur bouche": mouthWidth > 0 ? lipHeight / mouthWidth : 0,
  };

  const targets: Record<string, number> = {
    "Rapport Longueur/ Largeur du visage": 1.618,
    "Distance inter-oculaire / Largeur bouche": 1.618,
    "Distance inter-oculaire / Largeur visage": 0.32,
    "Nez→Menton / Longueur visage": 0.618,
    "Hauteur lèvres / Largeur bouche": 0.2,
  };

  const relErrors = Object.keys(ratios).map((k) => {
    const ideal = targets[k];
    const val = ratios[k];
    return ideal > 0 ? Math.abs(val - ideal) / ideal : 1;
  });

  const avgDev = relErrors.reduce((a, b) => a + b, 0) / relErrors.length;
  const score = Math.round(100 * Math.exp(-5 * avgDev) * 100) / 100;

  const values: Record<string, number | Record<string, number>> = {
    "Longueur du visage": round2(faceLength),
    "Largeur du visage": round2(faceWidth),
    "Distance inter-oculaire": round2(eyeDist),
    "Largeur de la bouche": round2(mouthWidth),
    "Nez→Menton": round2(noseToChin),
    "Hauteur des lèvres": round2(lipHeight),
    "Score Nombre d’Or": score,
    "Ratios évalués": Object.fromEntries(Object.entries(ratios).map(([k, v]) => [k, round3(v)])),
  };

  return { score, values, faceWidth, eyeDist, eyeL, eyeR, ratios, targets };
}

function computeOverall(symmetry: number, golden: number, faceWidth: number, eyeDist: number) {
  let base = 0.4 * symmetry + 0.4 * golden;
  let eyeScore = 75;
  let eyeSpacingRatio = 0;

  if (faceWidth > 0) {
    eyeSpacingRatio = eyeDist / faceWidth;
    const es = METHODOLOGY.eyeSpacing;
    if (eyeSpacingRatio > es.idealMin && eyeSpacingRatio < es.idealMax) eyeScore = 100;
    else if (eyeSpacingRatio <= es.low || eyeSpacingRatio >= es.high) eyeScore = 60;
    else eyeScore = 80;
  }
  base += 0.2 * eyeScore;

  const harmony = clamp(base, 20, 100);
  const weighted = METHODOLOGY.weights.symmetry * symmetry + METHODOLOGY.weights.golden * golden + METHODOLOGY.weights.harmony * harmony;
  const raw = weighted / 100;
  const scaled = Math.pow(raw, 1.8);
  const overall = clamp(4 + scaled * 9, 1, 10);

  const asymFactor = Math.abs(50 - symmetry) / 50;
  const goldenDev = Math.abs(60 - golden) / 60;
  const harmVar = Math.abs(70 - harmony) / 70;
  const uniqRaw = 0.4 * asymFactor + 0.3 * goldenDev + 0.3 * harmVar;
  const uniqueness = clamp(5 + 4 * uniqRaw, 1, 10);

  const fb = buildFeedback(symmetry, golden, eyeSpacingRatio);
  return { harmony: round2(harmony), overall: round1(overall), uniqueness: round1(uniqueness), feedback: fb };
}

function buildFeedback(symmetry: number, golden: number, eyeSpacingRatio: number) {
  const lines: string[] = [];
  if (symmetry > 90) lines.push("Symétrie faciale très élevée — indicateur de régularité morphologique.");
  else if (symmetry > 75) lines.push("Symétrie globalement bonne avec de légères asymétries naturelles.");
  else lines.push("Asymétries plus marquées — fréquentes et non pathologiques.");

  if (golden > 90) lines.push("Proportions très proches du nombre d’or.");
  else if (golden > 75) lines.push("Légères déviations par rapport au nombre d’or, sans conséquence esthétique directe.");
  else lines.push("Proportions éloignées du nombre d’or — rappel : la beauté ne se réduit pas à un ratio.");

  if (eyeSpacingRatio > 0) {
    const es = METHODOLOGY.eyeSpacing;
    if (eyeSpacingRatio > es.idealMin && eyeSpacingRatio < es.idealMax) lines.push("Espacement inter-oculaire dans une plage considérée harmonieuse.");
    else if (eyeSpacingRatio <= es.low) lines.push("Espacement inter-oculaire relativement réduit (aspect plus concentré).");
    else if (eyeSpacingRatio >= es.high) lines.push("Espacement inter-oculaire relativement large (impression plus ouverte).");
  }

  lines.push("Remarque : ces indicateurs sont descriptifs. La perception esthétique reste multidimensionnelle.");
  return lines.join("\n");
}

function getImageData(el: HTMLImageElement) {
  const c = document.createElement("canvas");
  c.width = el.naturalWidth; c.height = el.naturalHeight;
  const g = c.getContext("2d")!;
  g.drawImage(el, 0, 0);
  return g.getImageData(0, 0, c.width, c.height);
}
function qualityChecks(img: HTMLImageElement) {
  const gray = getImageData(img);
  const { data, width, height } = gray;

  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = (y * width + x) * 4;
      const dx = data[i + 0] - data[i - 4];
      const dy = data[i + 0] - data[i - width * 4];
      const g = Math.abs(dx) + Math.abs(dy);
      sum += g; sum2 += g * g; n++;
    }
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;

  let lumSum = 0;
  for (let i = 0; i < data.length; i += 4) lumSum += data[i];
  const luminance = lumSum / (data.length / 4);

  return {
    blurOK: variance > METHODOLOGY.quality.blurVarMin,
    exposureOK: luminance > METHODOLOGY.quality.luminanceMin && luminance < METHODOLOGY.quality.luminanceMax,
    variance,
    luminance,
  };
}

function alignAndCrop(img: HTMLImageElement, eyeL: [number, number], eyeR: [number, number]) {
  const angle = Math.atan2(eyeR[1] - eyeL[1], eyeR[0] - eyeL[0]);
  const w = img.naturalWidth, h = img.naturalHeight;

  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext("2d")!;
  tctx.translate(w / 2, h / 2);
  tctx.rotate(-angle);
  tctx.drawImage(img, -w / 2, -h / 2);

  const cx = (eyeL[0] + eyeR[0]) / 2;
  const cy = (eyeL[1] + eyeR[1]) / 2;
  const d = Math.hypot(eyeR[0] - eyeL[0], eyeR[1] - eyeL[1]);
  const size = d * 5.0;
  const x = clamp(cx - size / 2, 0, w - size);
  const y = clamp(cy - size * 0.6, 0, h - size * 1.1);
  const bw = Math.min(w - x, size);
  const bh = Math.min(h - y, size * 1.1);

  const canvas = document.createElement("canvas");
  canvas.width = 800;
  canvas.height = Math.round(800 * (bh / bw));
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(tmp, x, y, bw, bh, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function estimateStableLandmarks(
  detector: faceLandmarksDetection.FaceLandmarksDetector,
  video: HTMLVideoElement,
  frames = 7
) {
  const all: Array<Array<{ x: number; y: number }>> = [];
  for (let i = 0; i < frames; i++) {
    const preds = await detector.estimateFaces(video, { flipHorizontal: false });
    if (preds?.[0]) {
      const kps = preds[0].keypoints.map((k) => ({
        x: k.x / video.videoWidth,
        y: k.y / video.videoHeight,
      }));
      all.push(kps);
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  if (!all.length) return null;
  const nPts = all[0].length;
  const median: LM[] = [];
  for (let p = 0; p < nPts; p++) {
    const xs = all.map((a) => a[p].x).sort((a, b) => a - b);
    const ys = all.map((a) => a[p].y).sort((a, b) => a - b);
    const mid = Math.floor(xs.length / 2);
    median.push({ x: xs[mid], y: ys[mid] });
  }
  return median;
}

function jitterLandmarks(lms: LM[], sigma: number) {
  const rnd = () => (Math.random() * 2 - 1) * sigma;
  return lms.map((p) => ({ x: clamp(p.x + rnd(), 0, 1), y: clamp(p.y + rnd(), 0, 1), z: p.z }));
}
function withUncertainty<T>(calcOnce: () => T & { scalar: number }, repeats: number) {
  const vals: number[] = [];
  for (let i = 0; i < repeats; i++) {
    vals.push(calcOnce().scalar);
  }
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vals.length);
  return { mean, sd, ci95: 1.96 * sd };
}

export default function App() {
  const [model, setModel] = useState<faceLandmarksDetection.FaceLandmarksDetector | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [processing, setProcessing] = useState(false);

  const [scores, setScores] = useState<{ symmetry?: number; golden?: number; harmony?: number; overall?: number; uniqueness?: number }>({});
  const [ci, setCi] = useState<{ symmetry?: number; golden?: number; harmony?: number; overall?: number; uniqueness?: number }>({});
  const [feedback, setFeedback] = useState<string>("");
  const [measures, setMeasures] = useState<Record<string, number | Record<string, number>>>({});
  const [ratios, setRatios] = useState<Record<string, number>>({});
  const [targets, setTargets] = useState<Record<string, number>>({});
  const [quality, setQuality] = useState<{ blurOK: boolean; exposureOK: boolean; variance: number; luminance: number } | null>(null);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasLeftRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRightRef = useRef<HTMLCanvasElement | null>(null);
  const canvasFullLeftRef = useRef<HTMLCanvasElement | null>(null);
  const canvasFullRightRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    (async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      const detector = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        { runtime: "tfjs", refineLandmarks: true }
      );
      try { await detector.estimateFaces(document.createElement("canvas"), { flipHorizontal: false }); } catch {}
      setModel(detector);
    })();
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const onSelectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileName(f.name);
    setImageURL(URL.createObjectURL(f));
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream as MediaStream;
        await videoRef.current.play();
      }
    } catch {
      alert("Impossible d'accéder à la caméra. Veuillez autoriser l'accès ou téléverser une image.");
    }
  };

  const takeSnapshot = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setFileName("capture.jpg");
    setImageURL(canvas.toDataURL("image/jpeg"));
  };

  const analyze = async () => {
    if (!model) { alert("Modèle non prêt."); return; }
    if (!imgRef.current) { alert("Aucune image chargée."); return; }

    setProcessing(true);
    try {
      const img = imgRef.current;

      const qc = qualityChecks(img);
      setQuality(qc);
      if (!qc.blurOK || !qc.exposureOK) {
        alert("Image trop floue ou mal exposée. Essayez une photo plus nette/éclairée.");
      }

      const firstPreds = await model.estimateFaces(img, { flipHorizontal: false });
      if (!firstPreds || firstPreds.length === 0) throw new Error("Aucun visage détecté. Fournissez une vue de face nette et centrée.");
      const kps0 = firstPreds[0].keypoints;
      const eyeL0: [number, number] = [kps0[133].x, kps0[133].y];
      const eyeR0: [number, number] = [kps0[362].x, kps0[362].y];

      const alignedCanvas = alignAndCrop(img, eyeL0, eyeR0);

      const secondPreds = await model.estimateFaces(alignedCanvas, { flipHorizontal: false });
      if (!secondPreds || secondPreds.length === 0) throw new Error("Le visage n’a pas pu être confirmé après alignement.");
      const useW = alignedCanvas.width;
      const useH = alignedCanvas.height;
      const baseLm: LM[] = secondPreds[0].keypoints.map((k: any) => ({ x: k.x / useW, y: k.y / useH }));

      const symmetryBase = calcSymmetryScore(baseLm);
      const gBase = analyzeGolden(useW, useH, baseLm);
      const overallBase = computeOverall(symmetryBase, gBase.score, gBase.faceWidth, gBase.eyeDist);

      const rep = METHODOLOGY.uncertainty.repeats;
      const sigma = METHODOLOGY.uncertainty.jitterSigma;

      const symCI = withUncertainty(() => {
        const lm = jitterLandmarks(baseLm, sigma);
        return { scalar: calcSymmetryScore(lm) };
      }, rep);

      const goldCI = withUncertainty(() => {
        const lm = jitterLandmarks(baseLm, sigma);
        return { scalar: analyzeGolden(useW, useH, lm).score };
      }, rep);

      const harmCI = withUncertainty(() => {
        const lm = jitterLandmarks(baseLm, sigma);
        const g = analyzeGolden(useW, useH, lm);
        const s = calcSymmetryScore(lm);
        return { scalar: computeOverall(s, g.score, g.faceWidth, g.eyeDist).harmony };
      }, rep);

      const overallCI = withUncertainty(() => {
        const lm = jitterLandmarks(baseLm, sigma);
        const g = analyzeGolden(useW, useH, lm);
        const s = calcSymmetryScore(lm);
        return { scalar: computeOverall(s, g.score, g.faceWidth, g.eyeDist).overall };
      }, rep);

      const uniqCI = withUncertainty(() => {
        const lm = jitterLandmarks(baseLm, sigma);
        const g = analyzeGolden(useW, useH, lm);
        const s = calcSymmetryScore(lm);
        return { scalar: computeOverall(s, g.score, g.faceWidth, g.eyeDist).uniqueness };
      }, rep);

      setScores({
        symmetry: round2(symmetryBase),
        golden: round2(gBase.score),
        harmony: overallBase.harmony,
        overall: overallBase.overall,
        uniqueness: overallBase.uniqueness,
      });
      setCi({
        symmetry: round2(symCI.ci95),
        golden: round2(goldCI.ci95),
        harmony: round2(harmCI.ci95),
        overall: round2(overallCI.ci95),
        uniqueness: round2(uniqCI.ci95),
      });
      setFeedback(overallBase.feedback);
      setMeasures(gBase.values);
      setRatios(gBase.ratios);
      setTargets(gBase.targets);

      renderProfilesFromCanvas(alignedCanvas);

    } catch (e: any) {
      alert(e?.message || String(e));
    } finally {
      setProcessing(false);
    }
  };

  function renderProfilesFromCanvas(source: HTMLCanvasElement) {
    if (!canvasLeftRef.current || !canvasRightRef.current || !canvasFullLeftRef.current || !canvasFullRightRef.current) return;

    const w = source.width, h = source.height;
    const cx = Math.floor(w / 2);

    const leftCtx = canvasLeftRef.current.getContext("2d");
    const rightCtx = canvasRightRef.current.getContext("2d");
    const fullLCtx = canvasFullLeftRef.current.getContext("2d");
    const fullRCtx = canvasFullRightRef.current.getContext("2d");
    if (!leftCtx || !rightCtx || !fullLCtx || !fullRCtx) return;

    [canvasLeftRef.current, canvasRightRef.current, canvasFullLeftRef.current, canvasFullRightRef.current].forEach((c) => {
      c.width = cx;
      c.height = h;
    });

    leftCtx.clearRect(0, 0, cx, h);
    rightCtx.clearRect(0, 0, cx, h);
    leftCtx.drawImage(source, 0, 0, cx, h, 0, 0, cx, h);
    rightCtx.drawImage(source, cx, 0, cx, h, 0, 0, cx, h);

    fullLCtx.clearRect(0, 0, cx, h);
    fullLCtx.drawImage(source, 0, 0, cx, h, 0, 0, cx, h);
    fullLCtx.save();
    fullLCtx.translate(cx, 0);
    fullLCtx.scale(-1, 1);
    fullLCtx.drawImage(source, 0, 0, cx, h, 0, 0, cx, h);
    fullLCtx.restore();

    fullRCtx.clearRect(0, 0, cx, h);
    fullRCtx.save();
    fullRCtx.scale(-1, 1);
    fullRCtx.drawImage(source, cx, 0, cx, h, -cx, 0, cx, h);
    fullRCtx.restore();
    fullRCtx.drawImage(source, cx, 0, cx, h, 0, 0, cx, h);
  }

  return (
    <div className="app-bg min-h-screen text-neutral-900">
      <div className="mx-auto max-w-6xl px-5 md:px-8 py-10 md:py-14">
        <header className="mb-10 md:mb-14">
          <div className="mb-2 text-[12px] uppercase tracking-[0.18em] text-neutral-600">Analyse faciale</div>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight leading-tight">Rapport personnalisé d’harmonie faciale</h1>
          <p className="mt-4 max-w-3xl text-[15px] leading-7 text-neutral-700">
            L’analyse repose sur des mesures géométriques de landmarks faciaux détectés localement dans votre navigateur.
            Les résultats sont descriptifs, comparés à des cibles, et assortis d’une incertitude statistique.
          </p>
        </header>

        <section className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardBody>
              <h2 className="text-lg font-medium">Import de l’image</h2>
              <p className="mt-2 text-sm text-neutral-700">
                Téléversez une photo de face, nette, correctement exposée, visage centré, expression neutre.
              </p>

              <div className="mt-5">
                <div className="file-field">
                  <input
                    ref={inputRef}
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={onSelectFile}
                    className="file-input"
                  />
                  <label htmlFor="file-input" className="file-label">
                    <span>Choisir un fichier</span>
                    <span className="file-name">{fileName || "Aucun fichier"}</span>
                  </label>
                </div>
              </div>

              <div className="mt-8 border-t border-neutral-200 pt-6">
                <h3 className="text-sm font-medium">Ou utiliser la caméra</h3>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <Button onClick={startCamera}>Activer la caméra</Button>
                  <Button variant="outline" onClick={takeSnapshot}>
                    Capturer
                  </Button>
                </div>
                <video ref={videoRef} className="mt-4 w-full rounded-xl border border-neutral-200" autoPlay muted playsInline />
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardBody>
              <h2 className="text-lg font-medium">Aperçu</h2>
              <div className="mt-4 rounded-xl border border-neutral-200 p-3 bg-white">
                {imageURL ? (
                  <img ref={imgRef} src={imageURL} alt="aperçu" className="mx-auto max-h-[460px] object-contain" />
                ) : (
                  <div className="h-48 grid place-items-center text-sm text-neutral-600">Aucune image sélectionnée.</div>
                )}
              </div>

              {quality && (
                <div className="mt-4 grid gap-2 text-sm">
                  <div className={`inline-flex items-center gap-2 ${quality.blurOK ? "text-emerald-700" : "text-red-700"}`}>
                    Netteté : {quality.blurOK ? "OK" : "Insuffisante"} <span className="text-neutral-500">(variance ≈ {Math.round(quality.variance)})</span>
                  </div>
                  <div className={`inline-flex items-center gap-2 ${quality.exposureOK ? "text-emerald-700" : "text-red-700"}`}>
                    Exposition : {quality.exposureOK ? "OK" : "À corriger"} <span className="text-neutral-500">(luminance ≈ {Math.round(quality.luminance)})</span>
                  </div>
                </div>
              )}

              <div className="mt-6">
                <Button size="lg" disabled={!imageURL || !model || processing} onClick={analyze}>
                  {processing ? "Analyse en cours…" : "Lancer l’analyse"}
                </Button>
              </div>
            </CardBody>
          </Card>
        </section>

        {scores.symmetry !== undefined && (
          <>
            <section className="mt-10 md:mt-14 grid gap-6 md:grid-cols-2">
              <Card>
                <CardBody>
                  <h2 className="text-lg font-medium">Résultats quantitatifs</h2>
                  <div className="mt-5 grid gap-5">
                    <Metric label="Score de symétrie" value={scores.symmetry!} ci={ci.symmetry} max={100} />
                    <Metric label="Concordance au nombre d’or" value={scores.golden!} ci={ci.golden} max={100} />
                    <Metric label="Score d’harmonie faciale" value={scores.harmony!} ci={ci.harmony} max={100} />
                    <Metric label="Indice global" value={scores.overall!} ci={ci.overall} max={10} />
                    <Metric label="Indice d’originalité" value={scores.uniqueness!} ci={ci.uniqueness} max={10} />
                  </div>
                </CardBody>
              </Card>

              <Card>
                <CardBody>
                  <h2 className="text-lg font-medium">Analyse et conclusions</h2>
                  <pre className="mt-4 whitespace-pre-wrap rounded-xl bg-neutral-50 border border-neutral-200 p-4 text-sm leading-6 text-neutral-800">
                    {feedback}
                  </pre>
                  <details className="group mt-5">
                    <summary className="cursor-pointer text-sm font-medium text-neutral-900">Contexte et limites</summary>
                    <p className="mt-2 text-sm text-neutral-700">
                      Indicateurs calculés à partir de landmarks faciaux (MediaPipe FaceMesh) et de ratios géométriques.
                      Les intervalles ±IC95 proviennent d’un bootstrap par jitter aléatoire sur les points détectés.
                    </p>
                  </details>
                </CardBody>
              </Card>
            </section>

            {/* Mesures Faciales détaillées */}
            <section className="mt-10 md:mt-14">
              <h2 className="text-lg font-medium">Mesures faciales</h2>
              <p className="mt-1 text-sm text-neutral-700">Distances exprimées en pixels sur l’image alignée.</p>
              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <Measure label="Longueur du visage" value={measures["Longueur du visage"]} unit="px" />
                <Measure label="Largeur du visage" value={measures["Largeur du visage"]} unit="px" />
                <Measure label="Distance inter-oculaire" value={measures["Distance inter-oculaire"]} unit="px" />
                <Measure label="Largeur de la bouche" value={measures["Largeur de la bouche"]} unit="px" />
                <Measure label="Nez→Menton" value={measures["Nez→Menton"]} unit="px" />
                <Measure label="Hauteur des lèvres" value={measures["Hauteur des lèvres"]} unit="px" />
                <Measure label="Score Nombre d’Or" value={measures["Score Nombre d’Or"]} unit="/100" />
              </div>
            </section>

            {/* Ratios détaillés avec cibles et erreur relative */}
            <section className="mt-10 md:mt-14">
              <h2 className="text-lg font-medium">Ratios évalués</h2>
              <p className="mt-1 text-sm text-neutral-700">Comparaison aux valeurs cibles et erreur relative.</p>
              <div className="mt-4 overflow-hidden rounded-xl border border-neutral-200 bg-white">
                <table className="w-full text-sm">
                  <thead className="bg-neutral-50 text-neutral-800">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">Ratio</th>
                      <th className="px-4 py-3 text-right font-medium">Mesuré</th>
                      <th className="px-4 py-3 text-right font-medium">Cible</th>
                      <th className="px-4 py-3 text-right font-medium">Erreur relative</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100">
                    {Object.keys(ratios).map((k) => {
                      const val = ratios[k];
                      const tgt = targets[k];
                      const rel = typeof tgt === "number" && tgt > 0 ? Math.abs(val - tgt) / tgt : 1;
                      return (
                        <tr key={k} className="hover:bg-neutral-50/60">
                          <td className="px-4 py-3 text-neutral-800">{k}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{round3(val)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{round3(tgt)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{round3(rel * 100)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="mt-10 md:mt-14">
              <h2 className="text-lg font-medium">Profils symétrisés</h2>
              <p className="mt-1 text-sm text-neutral-700">Visualisations générées à partir des moitiés gauche et droite (image alignée).</p>
              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-4">
                <CanvasFrame title="Profil gauche" canvasRef={canvasLeftRef} />
                <CanvasFrame title="Profil droit" canvasRef={canvasRightRef} />
                <CanvasFrame title="Visage complet (depuis le gauche)" canvasRef={canvasFullLeftRef} />
                <CanvasFrame title="Visage complet (depuis le droit)" canvasRef={canvasFullRightRef} />
              </div>
            </section>
          </>
        )}

        <footer className="mt-16 md:mt-20 border-t border-neutral-200 pt-6 text-xs text-neutral-600">
          <p>Traitement local dans votre navigateur, aucune image n’est envoyée sur un serveur.</p>
          <p className="mt-1">Méthodologie v{METHODOLOGY.version} — pondérations : sym {METHODOLOGY.weights.symmetry}, or {METHODOLOGY.weights.golden}, harmonie {METHODOLOGY.weights.harmony}.</p>
        </footer>
      </div>
    </div>
  );
}

function Metric({ label, value, max, ci }: { label: string; value: number; max: number; ci?: number }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="rounded-xl border border-neutral-200 p-4 bg-white/70">
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-neutral-700">{label}</span>
        <span className="text-sm font-semibold">
          {value}
          {typeof ci === "number" ? ` ± ${ci}` : ""}{max === 10 ? " / 10" : " / 100"}
        </span>
      </div>
      <Progress value={pct} className="mt-3" />
    </div>
  );
}

function Measure({ label, value, unit }: { label: string; value: number | Record<string, number> | undefined; unit?: string }) {
  const v = typeof value === "number" ? value : undefined;
  return (
    <div className="rounded-xl border border-neutral-200 p-4 bg-white/70">
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-neutral-700">{label}</span>
        <span className="text-sm font-semibold">{typeof v === "number" ? `${round2(v)}${unit ? " " + unit : ""}` : "—"}</span>
      </div>
    </div>
  );
}

function CanvasFrame({ title, canvasRef }: { title: string; canvasRef: React.RefObject<HTMLCanvasElement> }) {
  return (
    <Card>
      <CardBody className="p-4">
        <div className="text-sm font-medium">{title}</div>
        <div className="mt-3 overflow-hidden rounded-xl border border-neutral-200 bg-white">
          <canvas ref={canvasRef} className="block w-full" />
        </div>
      </CardBody>
    </Card>
  );
}
