import React, { memo, useCallback, useState } from 'react';
import { Copy } from './Icons';

interface PromptViewerProps {
    content: string | null;
    fileName: string | null;
    projectName?: string;
    onEdit?: (content: string) => void;
}

export const PromptViewer = memo(({ content, fileName, projectName, onEdit }: PromptViewerProps) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(() => {
        if (content) {
            navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 800);
        }
    }, [content]);

    if (content === null) {
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
                    <span className="text-[10px] font-display font-bold uppercase tracking-[0.15em] text-foreground/40">
                        {projectName}
                    </span>
                    <span className="font-mono text-[11px] font-medium text-foreground/60 tracking-tight">{fileName}</span>
                </div>

                <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-blue-500/40 mr-2">Editable Mode</span>
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
            </div>

            {/* Content Area with editable textarea */}
            <div className="flex-1 overflow-hidden p-0 relative">
                <div key={fileName} className="h-full w-full animate-prompt-entry">
                    <textarea
                        value={content}
                        onChange={(e) => onEdit?.(e.target.value)}
                        className="w-full h-full p-12 bg-transparent resize-none font-mono text-[13.5px] leading-[1.8] text-foreground/85 font-normal tracking-[-0.01em] outline-none border-none selection:bg-blue-100/50"
                        spellCheck={false}
                    />
                </div>
            </div>
        </div>
    );
});

PromptViewer.displayName = 'PromptViewer';
