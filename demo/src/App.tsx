import React, { useState, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Loader2, BookOpen, Layers, Zap, Database, ExternalLink, Activity } from "lucide-react";

// API Configuration
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Result {
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, any>;
}

interface SearchResponse {
  query: string;
  results: Result[];
  latency_ms: number;
  total_docs: number;
  system_metrics: {
    duration: number;
    user_cpu: number;
    sys_cpu: number;
    peak_heap_mb: number;
    peak_rss_mb: number;
  };
  quantization_mode: string;
}

const App: React.FC = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [totalDocs, setTotalDocs] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [useRerank, setUseRerank] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<SearchResponse["system_metrics"] | null>(null);
  const [quantization, setQuantization] = useState<string>("none");

  // Fetch initial stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await axios.get(`${API_URL}/health`);
        if (res.data.engine_ready) {
           // Initial check
        }
      } catch (err) {
        console.error("Failed to connect to backend", err);
      }
    };
    fetchStats();
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setHasSearched(true);
    try {
      const res = await axios.get<SearchResponse>(`${API_URL}/search`, {
        params: { q: query, top_k: 10, rerank: useRerank },
      });
      setResults(res.data.results);
      setLatency(res.data.latency_ms);
      setTotalDocs(res.data.total_docs);
      setSystemMetrics(res.data.system_metrics);
      setQuantization(res.data.quantization_mode);
    } catch (err) {
      console.error("Search failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative flex flex-col items-center pb-20">
      <div className="gradient-bg" />
      
      <main className="w-full max-w-5xl px-6 pt-20 pb-12 flex flex-col items-center">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8 text-primary shadow-lg border border-primary/20">
            <Activity size={18} />
            <span className="font-semibold text-xs tracking-widest uppercase">Deep Systems Profiling Enabled</span>
          </div>
          <h1 className="text-6xl font-bold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
            IRKit Discovery
          </h1>
          <p className="text-text-muted text-lg max-w-xl mx-auto">
            High-performance hybrid retrieval with hardware-level observability and vector quantization.
          </p>
        </motion.div>

        <div className="w-full flex flex-col lg:flex-row gap-8">
          {/* Left Column: Search & Results */}
          <div className="flex-1 flex flex-col items-center">
            {/* Search Bar */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="w-full relative mb-12"
            >
              <div className="relative glass-card p-2 group">
                <div className="absolute left-6 top-1/2 -translate-y-1/2 text-text-muted group-focus-within:text-primary transition-colors">
                  <Search size={24} />
                </div>
                <input 
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder="Search research papers, abstracts..."
                  className="w-full h-16 pl-16 pr-32 bg-transparent border-none text-white text-xl placeholder:text-text-muted/50 focus:outline-none"
                />
                <button 
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-3 top-1/2 -translate-y-1/2 h-12 px-8 btn-primary rounded-xl font-bold flex items-center gap-2"
                >
                  {loading ? <Loader2 className="animate-spin" /> : "Search"}
                </button>
              </div>

              {/* Toggles */}
              <div className="mt-4 flex justify-between items-center px-2">
                <div className="flex items-center gap-2">
                   <div className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${quantization !== 'none' ? 'bg-green-500/20 text-green-400' : 'bg-slate-800 text-slate-500'}`}>
                     Mode: {quantization}
                   </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${useRerank ? 'text-primary' : 'text-text-muted'}`}>
                    Reranking
                  </span>
                  <button 
                    onClick={() => setUseRerank(!useRerank)}
                    className={`w-10 h-5 rounded-full relative transition-colors ${useRerank ? 'bg-primary' : 'bg-slate-700'}`}
                  >
                    <motion.div 
                      animate={{ x: useRerank ? 22 : 4 }}
                      className="absolute top-1 w-3 h-3 bg-white rounded-full shadow-md"
                    />
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Results */}
            <div className="w-full space-y-6">
              <AnimatePresence mode="popLayout">
                {loading ? (
                  <motion.div 
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center py-20"
                  >
                     <Loader2 size={48} className="text-primary animate-spin mb-4" />
                     <p className="text-text-muted">Analyzing query vectors...</p>
                  </motion.div>
                ) : results.length > 0 ? (
                  results.map((res, idx) => (
                    <motion.div 
                      key={res.doc_id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="glass-card p-6 result-card group overflow-hidden relative"
                    >
                      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 -mr-16 -mt-16 rounded-full blur-3xl group-hover:bg-primary/10 transition-colors" />
                      
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="text-xl font-bold text-white group-hover:text-primary transition-colors pr-8">
                           {res.title}
                        </h3>
                        <a 
                          href={res.metadata.url || `https://arxiv.org/abs/${res.doc_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-text-muted hover:text-white transition-colors p-2"
                        >
                          <ExternalLink size={18} />
                        </a>
                      </div>
                      
                      <p className="text-text-muted leading-relaxed mb-4 line-clamp-3">
                        {res.snippet}
                      </p>
                      
                      <div className="flex flex-wrap items-center gap-4 text-xs font-semibold uppercase tracking-wider">
                        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-primary/20 text-primary border border-primary/20">
                          <Zap size={10} />
                          Score: {res.score.toFixed(4)}
                        </div>
                        {res.metadata.categories && (
                          <div className="px-2.5 py-1 rounded bg-slate-800 text-slate-400 border border-white/5">
                            {res.metadata.categories}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))
                ) : hasSearched && !loading ? (
                  <div className="text-center py-20 glass-card">
                    <h3 className="text-xl font-bold">No results found</h3>
                  </div>
                ) : (
                  !hasSearched && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                       {[
                         { q: "Attention Mechanisms", desc: "Transformers & Self-Attention" },
                         { q: "Deep Reinforcement Learning", desc: "Policy Gradients & Q-Learning" },
                         { q: "Generative Adversarial Networks", desc: "Image Synthesis & GANs" },
                         { q: "Natural Language Processing", desc: "LLMs & Vector Semantics" }
                       ].map((item, i) => (
                         <button 
                           key={i}
                           onClick={() => { setQuery(item.q); handleSearch(); }}
                           className="glass-card p-6 text-left hover:bg-white/5 transition-colors group"
                         >
                           <div className="text-primary mb-2 group-hover:scale-110 transition-transform inline-block">
                             <Zap size={20} />
                           </div>
                           <div className="font-bold text-lg mb-1">{item.q}</div>
                           <div className="text-text-muted text-sm">{item.desc}</div>
                         </button>
                       ))}
                    </div>
                  )
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Right Column: System Insights */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-full lg:w-80 flex flex-col gap-6"
          >
            <div className="glass-card p-6 border-l-4 border-primary">
              <div className="flex items-center gap-2 mb-6">
                <Activity size={20} className="text-primary" />
                <h2 className="font-bold text-lg">System Insights</h2>
              </div>

              <div className="space-y-6">
                {/* Latency Wrapper */}
                <div>
                  <div className="flex justify-between text-xs font-bold text-text-muted uppercase mb-2">
                    <span>Search Latency</span>
                    <span className="text-primary">{latency ? latency.toFixed(2) : "0"}ms</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: latency ? Math.min(latency, 100) : 0 }}
                      className="bg-primary h-full"
                    />
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-1 gap-4">
                  <div className="p-3 rounded bg-slate-900/50 border border-white/5">
                    <div className="text-[10px] text-text-muted uppercase font-bold mb-1">User CPU Time</div>
                    <div className="text-lg font-mono text-white">{systemMetrics ? systemMetrics.user_cpu.toFixed(6) : "0.000"}s</div>
                  </div>
                  <div className="p-3 rounded bg-slate-900/50 border border-white/5">
                    <div className="text-[10px] text-text-muted uppercase font-bold mb-1">Peak Heap Memory</div>
                    <div className="text-lg font-mono text-white">{systemMetrics ? systemMetrics.peak_heap_mb.toFixed(2) : "0.00"} MB</div>
                  </div>
                  <div className="p-3 rounded bg-slate-900/50 border border-white/5">
                    <div className="text-[10px] text-text-muted uppercase font-bold mb-1">Peak RSS (OS)</div>
                    <div className="text-lg font-mono text-white">{systemMetrics ? systemMetrics.peak_rss_mb.toFixed(2) : "0.00"} MB</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-primary/5 border border-primary/10">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap size={16} className="text-primary" />
                    <span className="font-bold text-sm">Optimization</span>
                  </div>
                  <p className="text-xs text-text-muted leading-relaxed">
                    {quantization === 'pq' 
                      ? "PQ (Product Quantization) is active. Using Asymmetric Distance Computation (ADC) for O(1) distance lookups."
                      : quantization === 'sq8'
                      ? "SQ8 (Scalar Quantization) is active. Vectors are compressed to 8-bit integers (4x reduction)."
                      : "Standard FP32 mode. Large vector buffers are being processed directly by FAISS."}
                  </p>
                </div>
              </div>
            </div>

            <div className="glass-card p-6">
               <div className="flex items-center gap-2 mb-4">
                 <Database size={18} className="text-blue-400" />
                 <h2 className="font-bold text-sm">Engine Status</h2>
               </div>
               <div className="text-2xl font-bold mb-1">{totalDocs.toLocaleString()}</div>
               <div className="text-xs text-text-muted uppercase font-bold tracking-wider">Documents In Memory</div>
            </div>
          </motion.div>
        </div>
      </main>
      
      <footer className="w-full max-w-5xl px-6 py-12 text-center text-text-muted text-sm border-t border-white/5">
        <p>© 2026 IRKit — Advanced Hybrid Information Retrieval Portfolio Project</p>
      </footer>
    </div>
  );
};

export default App;
