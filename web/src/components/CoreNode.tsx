import React, { memo } from 'react';
import { ChevronRight } from './Icons';

interface CoreNodeProps {
    title: string;
    isExpanded: boolean;
    onToggle: () => void;
}

export const CoreNode = memo(({ title, isExpanded, onToggle }: CoreNodeProps) => {
    return (
        <div className="flex flex-col items-center w-full max-w-xl mx-auto">
            <div className="flex items-center gap-3 w-full p-4 bg-node rounded-lg border border-black/[0.03] transition-colors duration-160">
                <button
                    onClick={onToggle}
                    className={`p-1 hover:bg-black/[0.02] rounded-md transition-transform duration-160 [transition-timing-function:cubic-bezier(0.4,0,0.2,1)] ${isExpanded ? 'rotate-90' : ''}`}
                    aria-label={isExpanded ? "Collapse project" : "Expand project"}
                >
                    <ChevronRight />
                </button>
                <span className="font-semibold text-[18px] tracking-tight text-foreground">{title}</span>
            </div>
            {isExpanded && (
                <div className="w-px h-10 bg-black/[0.06]" />
            )}
        </div>
    );
});

CoreNode.displayName = 'CoreNode';
