"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, Mic } from "@/components/Icons";

export default function Home() {
  const [idea, setIdea] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!idea.trim() || isSubmitting) return;

    setIsSubmitting(true);
    try {
      const res = await fetch("/api/project", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idea }),
      });
      if (!res.ok) throw new Error("Failed to synthesize project");
      const data = await res.json();
      router.push(`/project/${data.id}`);
    } catch (err) {
      console.error("Failed to create project", err);
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-background">
      <div className={`w-full max-w-2xl flex flex-col items-center space-y-12 transition-all duration-700 ${isSubmitting ? 'opacity-20 scale-95 blur-sm grayscale' : 'opacity-100 scale-100 blur-0 grayscale-0'}`}>
        <h1 className="text-4xl font-medium tracking-tight text-foreground text-center">
          What are we building today?
        </h1>

        <form onSubmit={handleSubmit} className="w-full relative group">
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-foreground/20">
            <Plus />
          </div>

          <input
            type="text"
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            placeholder="Describe your project idea..."
            className="w-full h-16 pl-12 pr-12 bg-panel rounded-full border border-black/5 shadow-sm outline-none focus:border-black/10 focus:ring-4 focus:ring-black/[0.02] transition-all text-lg font-normal placeholder:text-foreground/20"
            disabled={isSubmitting}
            autoFocus
          />

          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-foreground/20">
            <Mic />
          </div>
        </form>

        <p className="text-xs font-medium text-foreground/30 uppercase tracking-widest">
          Deterministic Instruction Synthesis
        </p>
      </div>

      {isSubmitting && (
        <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 animate-in fade-in zoom-in duration-500">
          <div className="w-12 h-12 border-2 border-black/5 border-t-black/40 rounded-full animate-spin" />
          <div className="text-center space-y-1">
            <p className="text-sm font-medium text-foreground/60">Synthesizing Architecture</p>
            <p className="text-[10px] uppercase tracking-widest text-foreground/20 font-bold">Please wait...</p>
          </div>
        </div>
      )}
    </main>
  );
}
