"use client";

import React, { useEffect, useState } from 'react';
import { notFound, useParams } from 'next/navigation';
import { Project } from '@/lib/types';
import { ExplorerLayout } from '@/components/ExplorerLayout';

export default function ProjectPage() {
    const params = useParams();
    const id = params.id as string;
    const [project, setProject] = useState<Project | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchProject = async () => {
            try {
                const res = await fetch(`/api/project/${id}`);
                if (!res.ok) {
                    if (res.status === 404) notFound();
                    throw new Error('Failed to fetch project');
                }
                const data = await res.json();
                setProject(data);
            } catch (err) {
                console.error(err);
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setLoading(false);
            }
        };

        if (id) fetchProject();
    }, [id]);

    if (loading) {
        return (
            <main className="h-screen w-full bg-background flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-8 h-8 border-2 border-black/5 border-t-black/20 rounded-full animate-spin" />
                    <span className="text-xs font-medium text-foreground/30 uppercase tracking-widest">
                        Loading Synthesis
                    </span>
                </div>
            </main>
        );
    }

    if (error || !project) {
        return (
            <main className="h-screen w-full bg-background flex items-center justify-center p-6 text-center">
                <div className="max-w-md space-y-4">
                    <h2 className="text-xl font-medium text-foreground">Synthesis Error</h2>
                    <p className="text-foreground/60">{error || 'Could not load project.'}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-6 py-2 bg-panel border border-black/5 rounded-full text-sm font-medium hover:bg-black/[0.02] transition-colors"
                    >
                        Try Again
                    </button>
                </div>
            </main>
        );
    }

    return (
        <main className="h-screen w-full bg-background overflow-hidden animate-in fade-in duration-500">
            <ExplorerLayout project={project} />
        </main>
    );
}
