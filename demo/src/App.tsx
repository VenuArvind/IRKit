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
}

const App: React.FC = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [totalDocs, setTotalDocs] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Fetch initial stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await axios.get(`${API_URL}/health`);
        if (res.data.engine_ready) {
           // We could fetch more stats here if we had a dedicated endpoint
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
        params: { q: query, top_k: 10 },
      });
      setResults(res.data.results);
      setLatency(res.data.latency_ms);
      setTotalDocs(res.data.total_docs);
    } catch (err) {
      console.error("Search failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative flex flex-col items-center">
      <div className="gradient-bg" />
      
      <main className="w-full max-w-4xl px-6 pt-20 pb-12 flex flex-col items-center">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-6 mb-8 text-primary shadow-lg">
            <Layers size={20} />
            <span className="font-semibold text-sm tracking-widest uppercase">Hybrid IR Engine</span>
          </div>
          <h1 className="text-6xl font-bold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
            IRKit Search
          </h1>
          <p className="text-text-muted text-lg max-w-xl mx-auto">
            Experience state-of-the-art hybrid information retrieval. Powered by BM25, Semantic FAISS, and RRF Fusion.
          </p>
        </motion.div>

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
          
          {/* Engine Stats */}
          <div className="mt-4 flex justify-center gap-8 text-sm text-text-muted">
             {latency !== null && (
               <div className="flex items-center gap-2">
                 <Zap size={14} className="text-yellow-400" />
                 <span>{latency.toFixed(2)}ms latency</span>
               </div>
             )}
             {totalDocs > 0 && (
               <div className="flex items-center gap-2">
                 <Database size={14} className="text-blue-400" />
                 <span>{totalDocs} docs indexed</span>
               </div>
             )}
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
                    {res.metadata.authors && (
                      <div className="text-text-muted flex items-center gap-1.5 lowercase">
                        <Activity size={12} />
                        {res.metadata.authors}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))
            ) : hasSearched && !loading ? (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-20 glass-card"
              >
                <div className="inline-flex p-4 rounded-full bg-slate-800 mb-4">
                   <BookOpen size={32} className="text-text-muted" />
                </div>
                <h3 className="text-xl font-bold mb-2">No matching papers found</h3>
                <p className="text-text-muted">Try refining your search or using broader terms.</p>
              </motion.div>
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
      </main>
      
      <footer className="w-full max-w-4xl px-6 py-12 text-center text-text-muted text-sm border-t border-white/5">
        <p>© 2026 IRKit — Advanced Hybrid Information Retrieval Portfolio Project</p>
      </footer>
    </div>
  );
};

export default App;
