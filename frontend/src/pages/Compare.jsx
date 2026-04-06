import { useState, useRef } from "react";
import Topbar from "./TopBar";
import API from "../api/api";

/* ── Metric meta ────────────────────────────────────────────── */
const METRIC_INFO = {
  psnr: {
    label: "PSNR",
    unit: "dB",
    description: "Peak Signal-to-Noise Ratio — higher is better. ≥40 dB is excellent.",
    colorVar: "--metric-psnr",
  },
  mse: {
    label: "MSE",
    unit: "",
    description: "Mean Squared Error — lower is better. 0 means identical.",
    colorVar: "--metric-mse",
  },
  nc: {
    label: "NC",
    unit: "",
    description: "Normalized Cross-Correlation — closer to 1.0 is better.",
    colorVar: "--metric-nc",
  },
  ssim: {
    label: "SSIM",
    unit: "",
    description: "Structural Similarity Index — closer to 1.0 is better.",
    colorVar: "--metric-ssim",
  },
  ber: {
    label: "BER",
    unit: "",
    description: "Bit Error Rate — lower is better. 0 means no bit errors.",
    colorVar: "--metric-ber",
  },
};

/* ── Quality badge ──────────────────────────────────────────── */
function getQualityBadge(metrics) {
  if (!metrics) return null;
  const { psnr, ssim } = metrics;
  if (psnr >= 40 && ssim >= 0.98) return { text: "Excellent", cls: "badge-excellent" };
  if (psnr >= 30 && ssim >= 0.90) return { text: "Good", cls: "badge-good" };
  if (psnr >= 20 && ssim >= 0.70) return { text: "Fair", cls: "badge-fair" };
  return { text: "Poor", cls: "badge-poor" };
}

/* ── Ring progress for 0-1 metrics ──────────────────────────── */
function RingGauge({ value, size = 80, stroke = 6, color }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(value, 1));

  return (
    <svg width={size} height={size} className="ring-gauge">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="rgba(16,32,38,0.08)"
        strokeWidth={stroke}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" }}
      />
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
export default function Compare() {
  const [originalImage, setOriginalImage] = useState(null);
  const [watermarkedImage, setWatermarkedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState(null);

  const origRef = useRef(null);
  const wmRef = useRef(null);

  const handleCompare = async () => {
    setError(null);
    setMetrics(null);
    setDimensions(null);

    if (!originalImage) {
      setError("Please upload the original host image.");
      return;
    }
    if (!watermarkedImage) {
      setError("Please upload the watermarked image.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("original_image", originalImage);
      formData.append("watermarked_image", watermarkedImage);

      const res = await API.post("watermarking/compare/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setMetrics({
        psnr: res.data.psnr,
        mse: res.data.mse,
        nc: res.data.nc,
        ssim: res.data.ssim,
        ber: res.data.ber,
      });
      setDimensions(res.data.dimensions);
    } catch (err) {
      console.error(err);
      const msg = err?.response?.data?.error || "Comparison failed.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setWatermarkedImage(null);
    setMetrics(null);
    setError(null);
    setDimensions(null);
    setLoading(false);
  };

  const badge = getQualityBadge(metrics);

  return (
    <div className="page">
      <Topbar />

      <div className="content">
        <div className="compare-wrapper">
          <h2 className="page-title">Compare Quality</h2>
          <p className="subtitle center">
            Upload the original host image and its watermarked version to
            evaluate image quality metrics.
          </p>

          {/* ── Upload card ─────────────────────────────── */}
          <div className="auth-card reveal compare-card">
            <div className="upload-grid">
              {/* Original */}
              <div className="cmp-upload-section">
                <label className="vu-label">Original Host Image</label>
                {!originalImage ? (
                  <label className="upload-card cmp-drop" ref={origRef}>
                    <div className="cmp-drop-inner">
                      <span>Upload Original</span>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      hidden
                      onChange={(e) => setOriginalImage(e.target.files[0])}
                    />
                  </label>
                ) : (
                  <div className="preview-box cmp-preview">
                    <img src={URL.createObjectURL(originalImage)} alt="original" />
                    <button
                      className="remove-btn"
                      onClick={() => {
                        setOriginalImage(null);
                        setMetrics(null);
                      }}
                    >
                      ✕
                    </button>
                    <span className="cmp-tag">Original</span>
                  </div>
                )}
              </div>

              {/* Watermarked */}
              <div className="cmp-upload-section">
                <label className="vu-label">Watermarked Image</label>
                {!watermarkedImage ? (
                  <label className="upload-card cmp-drop" ref={wmRef}>
                    <div className="cmp-drop-inner">
                      <span>Upload Watermarked</span>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      hidden
                      onChange={(e) => setWatermarkedImage(e.target.files[0])}
                    />
                  </label>
                ) : (
                  <div className="preview-box cmp-preview">
                    <img src={URL.createObjectURL(watermarkedImage)} alt="watermarked" />
                    <button
                      className="remove-btn"
                      onClick={() => {
                        setWatermarkedImage(null);
                        setMetrics(null);
                      }}
                    >
                      ✕
                    </button>
                    <span className="cmp-tag wm">Watermarked</span>
                  </div>
                )}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="status error" style={{ marginTop: 14, fontSize: "0.82rem" }}>
                {error}
              </div>
            )}

            {/* Action buttons */}
            <div className="cmp-actions">
              <button
                className="primary-btn"
                onClick={handleCompare}
                disabled={loading}
              >
                {loading ? "Analyzing…" : "Run Comparison"}
              </button>
              {metrics && (
                <button className="ghost-btn" onClick={handleReset}>
                  Reset
                </button>
              )}
            </div>

            {loading && <div className="spinner" />}
          </div>

          {/* ── Results ────────────────────────────────── */}
          {metrics && (
            <div className="cmp-results reveal">
              {/* Quality badge */}
              {badge && (
                <div className={`cmp-quality-badge ${badge.cls}`}>
                  <span className="cmp-badge-dot" />
                  Overall Quality: <strong>{badge.text}</strong>
                </div>
              )}

              {/* Metric cards */}
              <div className="cmp-metric-grid">
                {Object.entries(METRIC_INFO).map(([key, info]) => {
                  const val = metrics[key];
                  const showRing = key === "ssim" || key === "nc";
                  const ringColor =
                    key === "ssim"
                      ? "rgba(15,118,110,0.85)"
                      : "rgba(59,130,246,0.85)";

                  return (
                    <div className="cmp-metric-card" key={key}>
                      <div className="cmp-metric-header">
                        <span className="cmp-metric-label">{info.label}</span>
                      </div>

                      <div className="cmp-metric-body">
                        {showRing && (
                          <div className="cmp-ring-wrap">
                            <RingGauge value={val} color={ringColor} />
                            <span className="cmp-ring-value">
                              {(val * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}

                        <span className="cmp-metric-value">
                          {key === "psnr" && val === Infinity
                            ? "∞"
                            : typeof val === "number"
                            ? val.toFixed(key === "mse" ? 4 : 6)
                            : val}
                          {info.unit && (
                            <span className="cmp-metric-unit"> {info.unit}</span>
                          )}
                        </span>
                      </div>

                      <p className="cmp-metric-desc">{info.description}</p>
                    </div>
                  );
                })}
              </div>

              {/* Dimension info */}
              {dimensions && (
                <div className="cmp-dim-bar">
                  <span>
                    Original: {dimensions.original[1]}×{dimensions.original[0]}px
                  </span>
                  <span className="cmp-dim-sep">|</span>
                  <span>
                    Watermarked: {dimensions.watermarked[1]}×
                    {dimensions.watermarked[0]}px
                  </span>
                  {dimensions.original[0] !== dimensions.watermarked[0] ||
                  dimensions.original[1] !== dimensions.watermarked[1] ? (
                    <span className="cmp-dim-warn">
                      ⚠ Images were resized to match
                    </span>
                  ) : null}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Scoped styles ────────────────────────────────── */}
      <style>{`
        .compare-wrapper {
          width: 100%;
          max-width: 720px;
          text-align: center;
        }

        .compare-card {
          width: 100%;
          max-width: 720px;
        }

        .cmp-upload-section {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .cmp-drop {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 160px;
          transition: border-color 0.25s, background 0.25s, transform 0.2s;
        }

        .cmp-drop:hover {
          transform: translateY(-2px);
        }

        .cmp-drop-inner {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          color: var(--ink-500);
          font: 500 0.92rem/1.3 "Space Grotesk", sans-serif;
        }

        .cmp-drop-icon {
          font-size: 2rem;
          line-height: 1;
        }

        .cmp-preview {
          position: relative;
        }

        .cmp-preview img {
          height: 180px;
        }

        .cmp-tag {
          position: absolute;
          bottom: 8px;
          left: 8px;
          background: rgba(15, 118, 110, 0.85);
          color: #fff;
          font: 600 0.68rem/1 "DM Mono", monospace;
          padding: 4px 8px;
          border-radius: 6px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        .cmp-tag.wm {
          background: rgba(139, 92, 246, 0.85);
        }

        .cmp-actions {
          margin-top: 24px;
          display: flex;
          justify-content: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        /* ── Results ──────────────────────────────────── */
        .cmp-results {
          margin-top: 28px;
        }

        .cmp-quality-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 20px;
          padding: 10px 20px;
          border-radius: 100px;
          font: 500 0.88rem/1.2 "Space Grotesk", sans-serif;
          border: 1.5px solid;
        }

        .cmp-badge-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .badge-excellent {
          background: rgba(16, 185, 129, 0.1);
          border-color: rgba(16, 185, 129, 0.4);
          color: #065f46;
        }
        .badge-excellent .cmp-badge-dot { background: #10b981; }

        .badge-good {
          background: rgba(59, 130, 246, 0.1);
          border-color: rgba(59, 130, 246, 0.4);
          color: #1e40af;
        }
        .badge-good .cmp-badge-dot { background: #3b82f6; }

        .badge-fair {
          background: rgba(245, 158, 11, 0.1);
          border-color: rgba(245, 158, 11, 0.4);
          color: #92400e;
        }
        .badge-fair .cmp-badge-dot { background: #f59e0b; }

        .badge-poor {
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.4);
          color: #991b1b;
        }
        .badge-poor .cmp-badge-dot { background: #ef4444; }

        /* ── Metric grid ─────────────────────────────── */
        .cmp-metric-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 14px;
        }

        .cmp-metric-card {
          border: 1px solid var(--line);
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.88);
          padding: 18px 16px 14px;
          text-align: left;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          backdrop-filter: blur(6px);
        }

        .cmp-metric-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 12px 28px rgba(11, 38, 46, 0.12);
        }

        .cmp-metric-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }

        .cmp-metric-icon {
          font-size: 1.3rem;
          line-height: 1;
        }

        .cmp-metric-label {
          font: 600 0.78rem/1 "DM Mono", monospace;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--ink-500);
        }

        .cmp-metric-body {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 10px;
        }

        .cmp-ring-wrap {
          position: relative;
          width: 80px;
          height: 80px;
          flex-shrink: 0;
        }

        .ring-gauge {
          display: block;
        }

        .cmp-ring-value {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          font: 700 0.82rem/1 "Space Grotesk", sans-serif;
          color: var(--ink-900);
        }

        .cmp-metric-value {
          font: 700 1.5rem/1.1 "Space Grotesk", sans-serif;
          color: var(--ink-900);
          word-break: break-all;
        }

        .cmp-metric-unit {
          font-size: 0.85rem;
          font-weight: 500;
          color: var(--ink-500);
        }

        .cmp-metric-desc {
          margin: 0;
          font: 400 0.74rem/1.45 "Space Grotesk", sans-serif;
          color: var(--ink-500);
        }

        /* ── Dimension bar ───────────────────────────── */
        .cmp-dim-bar {
          margin-top: 18px;
          padding: 10px 14px;
          border: 1px solid var(--line);
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.7);
          font: 500 0.78rem/1.4 "DM Mono", monospace;
          color: var(--ink-700);
          display: flex;
          justify-content: center;
          gap: 10px;
          flex-wrap: wrap;
        }

        .cmp-dim-sep {
          color: var(--line);
        }

        .cmp-dim-warn {
          color: #b45309;
          font-weight: 600;
        }

        /* ── Labels (reused from Verify/Extract) ──── */
        .vu-label {
          font: 600 0.78rem/1 "DM Mono", monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-500);
        }

        @media (max-width: 640px) {
          .cmp-metric-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
