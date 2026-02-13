import React, { memo, useCallback, useState } from 'react';
import { Copy } from './Icons';

interface PromptViewerProps {
    content: string | null;
    fileName: string | null;
    projectName?: string;
}

export const PromptViewer = memo(({ content, fileName, projectName }: PromptViewerProps) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(() => {
        if (content) {
            navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 800);
        }
    }, [content]);

    if (!content) {
        return (
            <div className="flex-1 bg-panel flex items-center justify-center text-foreground/20 font-medium tracking-tight">
                Select a file to view implementation prompt
            </div>
        );
    }

    return (
        <div className="flex-1 bg-panel flex flex-col h-full border-l border-black/[0.03]">
            {/* Header / Breadcrumb */}
            <div className="flex items-center justify-between p-5 border-b border-black/[0.03]">
                <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] font-medium uppercase tracking-[0.1em] text-foreground/30">
                        {projectName}
                    </span>
                    <span className="font-mono text-xs font-medium text-foreground/60">{fileName}</span>
                </div>

                <button
                    onClick={handleCopy}
                    className="p-2 hover:bg-black/[0.02] rounded-md flex items-center gap-2 transition-all duration-150 group"
                >
                    <Copy className={`transition-colors duration-200 ${copied ? 'text-blue-500' : 'text-foreground/30 group-hover:text-foreground/50'}`} />
                    <span className={`text-[11px] font-medium transition-all duration-200 ${copied ? 'text-blue-600 opacity-100' : 'text-foreground/40 opacity-0 group-hover:opacity-100'}`}>
                        {copied ? 'Copied' : 'Copy'}
                    </span>
                </button>
            </div>

            {/* Content Area with translation physics */}
            <div className="flex-1 overflow-auto p-10 scroll-smooth">
                <div key={fileName} className="max-w-3xl mx-auto animate-prompt-entry">
                    <pre className="font-mono text-[14px] leading-[1.7] whitespace-pre-wrap text-foreground/80 font-normal">
                        {content}
                    </pre>
                </div>
            </div>
        </div>
    );
});

PromptViewer.displayName = 'PromptViewer';
