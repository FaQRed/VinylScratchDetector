import React, { useState, useEffect, useRef, type ChangeEvent } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import {
  Upload, Activity, CheckCircle, AlertTriangle,
  Loader2, Play, Pause, Music, Disc, ChevronRight
} from 'lucide-react';

interface Scratch {
  second: number;
  probability: number;
}

interface AnalysisResponse {
  status: 'clean' | 'scratch_detected';
  total: number;
  seconds: Scratch[];
}

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>('rf');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!waveformRef.current || !file) return;

    if (wavesurfer.current) {
      wavesurfer.current.destroy();
    }

    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4f46e5',
      progressColor: '#818cf8',
      cursorColor: '#ffffff',
      barWidth: 3,
      barGap: 3,
      barRadius: 4,
      height: 200,
      normalize: true,
    });

    wavesurfer.current.load(URL.createObjectURL(file));
    wavesurfer.current.on('play', () => setIsPlaying(true));
    wavesurfer.current.on('pause', () => setIsPlaying(false));

    return () => wavesurfer.current?.destroy();
  }, [file]);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<AnalysisResponse>(
        `http://127.0.0.1:8000/analyze/${model}`,
        formData
      );
      setResult(response.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">
            <Disc size={28} className="rotating" />
          </div>
          <div>
            <h2>VinylScan</h2>
            <span>AI Audio Diagnostics</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-group">
            <label><Upload size={16} /> Source Audio</label>
            <div className="file-drop-zone">
              <input
                type="file"
                accept=".wav"
                onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] || null)}
              />
              <p>{file ? file.name : "Choose .wav file"}</p>
            </div>
          </div>

          <div className="nav-group">
            <label><Activity size={16} /> Engine Mode</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="rf">Random Forest</option>
              <option value="svm">SVM</option>
              <option value="cnn">CNN</option>
              <option value="mert">MERT</option>
            </select>
          </div>
        </nav>

        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className={`analyze-button ${loading ? 'loading' : ''}`}
        >
          {loading ? <Loader2 className="spinner" /> : <ChevronRight size={20} />}
          {loading ? "Analyzing..." : "Run Diagnostic"}
        </button>
      </aside>

      <main className="viewport">
        {file ? (
          <div className="workspace animate-fade-in">
            <div className="waveform-container shadow-premium">
              <div className="wf-header">
                <div className="file-info">
                  <Music size={18} />
                  <span>{file.name}</span>
                </div>
                <button className="play-toggle" onClick={() => wavesurfer.current?.playPause()}>
                  {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
                </button>
              </div>
              <div ref={waveformRef} />
            </div>

            <div className="results-area">
              {result && (
                <div className={`status-card ${result.status}`}>
                  <div className="status-icon">
                    {result.status === 'clean' ? <CheckCircle size={32} /> : <AlertTriangle size={32} />}
                  </div>
                  <div className="status-text">
                    <h3>{result.status === 'clean' ? 'Surface is Clean' : 'Defects Detected'}</h3>
                    <p>{result.status === 'clean' ? 'No significant scratches found.' : `Neural network identified ${result.total} anomalies.`}</p>
                  </div>
                </div>
              )}

              <div className="scratch-grid">
                {result?.seconds.map((s, i) => (
                  <div
                    key={i}
                    className="scratch-node"
                    onClick={() => wavesurfer.current?.setTime(s.second)}
                  >
                    <div className="node-header">
                      <span className="timestamp">{s.second.toFixed(1)}s</span>
                      <span className="badge">{(s.probability * 100).toFixed(0)}%</span>
                    </div>
                    <div className="progress-mini">
                      <div className="bar" style={{ width: `${s.probability * 100}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon-wrapper">
              <Disc size={80} strokeWidth={1} />
            </div>
            <h2>No Audio Loaded</h2>
            <p>Select a vinyl recording from the sidebar to begin AI analysis.</p>
          </div>
        )}
      </main>

      <style>{`
        :root {
          --bg-deep: #080a0f;
          --sidebar-color: #11141d;
          --card-color: #1a1f2e;
          --accent: #6366f1;
          --accent-hover: #818cf8;
          --text-p: #94a3b8;
          --success: #10b981;
          --danger: #ef4444;
        }

        body { margin: 0; background: var(--bg-deep); font-family: 'Inter', system-ui, sans-serif; }
        .app-shell { display: flex; width: 100vw; height: 100vh; color: white; overflow: hidden; }

        .sidebar {
          width: 340px;
          background: var(--sidebar-color);
          border-right: 1px solid rgba(255,255,255,0.05);
          padding: 40px 24px;
          display: flex;
          flex-direction: column;
          z-index: 10;
        }

        .logo-section { display: flex; align-items: center; gap: 15px; margin-bottom: 50px; }
        .logo-icon { 
          width: 48px; height: 48px; background: var(--accent); 
          border-radius: 12px; display: flex; align-items: center; justify-content: center;
          box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);
        }
        .logo-section h2 { margin: 0; font-size: 1.4rem; font-weight: 800; }
        .logo-section span { font-size: 0.75rem; color: var(--text-p); text-transform: uppercase; letter-spacing: 1px; }

        .nav-group { margin-bottom: 30px; }
        .nav-group label { display: flex; align-items: center; gap: 8px; color: var(--text-p); font-size: 0.85rem; font-weight: 600; margin-bottom: 12px; }
        
        .file-drop-zone {
          position: relative; border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px;
          padding: 20px; text-align: center; transition: 0.3s;
        }
        .file-drop-zone:hover { border-color: var(--accent); background: rgba(99, 102, 241, 0.05); }
        .file-drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .file-drop-zone p { margin: 0; font-size: 0.85rem; color: var(--text-p); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        select {
          width: 100%; background: #080a0f; border: 1px solid rgba(255,255,255,0.1);
          padding: 12px; border-radius: 10px; color: white; cursor: pointer; outline: none;
        }

        .analyze-button {
          margin-top: auto; background: var(--accent); border: none; padding: 18px;
          border-radius: 14px; color: white; font-weight: 700; font-size: 1rem;
          display: flex; align-items: center; justify-content: center; gap: 12px;
          cursor: pointer; transition: 0.3s;
          box-shadow: 0 10px 20px rgba(99, 102, 241, 0.2);
        }
        .analyze-button:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-2px); }
        .analyze-button:disabled { opacity: 0.4; filter: grayscale(1); }

        .viewport { flex: 1; position: relative; overflow-y: auto; background: radial-gradient(circle at top right, #111827, #080a0f); }
        .workspace { padding: 40px; max-width: 1100px; margin: 0 auto; }

        .waveform-container {
          background: var(--card-color); border-radius: 24px; padding: 30px; margin-bottom: 40px;
          border: 1px solid rgba(255,255,255,0.05);
        }
        .wf-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }
        .file-info { display: flex; align-items: center; gap: 10px; color: var(--text-p); font-size: 0.9rem; }
        .play-toggle {
          width: 60px; height: 60px; border-radius: 50%; border: none; 
          background: white; color: var(--bg-deep); cursor: pointer;
          display: flex; align-items: center; justify-content: center; transition: 0.2s;
        }
        .play-toggle:hover { transform: scale(1.1); box-shadow: 0 0 20px rgba(255,255,255,0.2); }

        .status-card { display: flex; align-items: center; gap: 20px; padding: 25px; border-radius: 20px; margin-bottom: 40px; }
        .status-card.clean { background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success); color: var(--success); }
        .status-card.scratch_detected { background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); color: var(--danger); }
        .status-text h3 { margin: 0 0 5px 0; font-size: 1.2rem; }
        .status-text p { margin: 0; font-size: 0.9rem; opacity: 0.8; color: white; }

        .scratch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 16px; }
        .scratch-node {
          background: var(--card-color); padding: 16px; border-radius: 16px; 
          border: 1px solid rgba(255,255,255,0.03); cursor: pointer; transition: 0.2s;
        }
        .scratch-node:hover { background: #252b3d; border-color: var(--accent); transform: translateY(-4px); }
        .node-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .timestamp { font-weight: 700; font-size: 1rem; }
        .badge { font-size: 0.7rem; padding: 2px 6px; background: rgba(255,255,255,0.05); border-radius: 4px; }
        .progress-mini { height: 4px; background: rgba(0,0,0,0.3); border-radius: 2px; overflow: hidden; }
        .progress-mini .bar { height: 100%; background: var(--accent); }

        .empty-state { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #1e293b; }
        .empty-icon-wrapper { margin-bottom: 20px; color: #1e293b; }
        .empty-state h2 { color: #475569; margin-bottom: 8px; }

        .spinner { animation: spin 1s linear infinite; }
        .rotating { animation: spin 3s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .animate-fade-in { animation: fadeIn 0.5s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
      `}</style>
    </div>
  );
};

export default App;