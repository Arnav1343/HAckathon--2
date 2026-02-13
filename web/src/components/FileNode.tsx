import React, { memo } from 'react';
import { ChevronRight } from './Icons';
import { FunctionContract } from '@/lib/types';

interface FunctionNodeProps {
    contract: FunctionContract;
    isLast: boolean;
}

export const FunctionNode = memo(({ contract, isLast }: FunctionNodeProps) => {
    const params = contract.parameters
        .map(p => `${p.name}: ${p.type}`)
        .join(', ');

    return (
        <div className="relative pl-14 py-1.5 flex items-center gap-3 font-mono text-[13px] font-normal text-foreground/50 group/fn">
            {/* Horizontal connect line */}
            <div className="absolute left-[24px] top-1/2 w-4 h-px bg-black/[0.08]" />

            <span className="transition-colors duration-150 group-hover/fn:text-foreground/80 lowercase">
                {contract.name}({params}) <span className="text-foreground/30">-&gt;</span> {contract.returnType}
            </span>
        </div>
    );
});

FunctionNode.displayName = 'FunctionNode';

interface FileNodeProps {
    path: string;
    isExpanded: boolean;
    isSelected: boolean;
    onToggle: (e: React.MouseEvent) => void;
    onSelect: () => void;
    isLast: boolean;
    children?: React.ReactNode;
}

export const FileNode = memo(({
    path,
    isExpanded,
    isSelected,
    onToggle,
    onSelect,
    isLast,
    children
}: FileNodeProps) => {
    return (
        <div className="flex flex-col w-full max-w-xl mx-auto relative group/file">
            {/* Hierarchical Vertical Line */}
            {!isLast && (
                <div className="absolute left-[24px] top-0 bottom-0 w-px bg-black/[0.06] z-0" />
            )}
            {isLast && (
                <div className="absolute left-[24px] top-0 h-4 w-px bg-black/[0.06] z-0" />
            )}

            <div className={`relative flex items-center gap-2 z-10 transition-all duration-160`}>
                <button
                    onClick={onToggle}
                    className={`p-1 hover:bg-black/[0.02] rounded-md transition-transform duration-160 [transition-timing-function:cubic-bezier(0.4,0,0.2,1)] ${isExpanded ? 'rotate-90 text-foreground/60' : 'text-foreground/30'}`}
                    aria-label={isExpanded ? "Collapse functions" : "Expand functions"}
                >
                    <ChevronRight />
                </button>
                <button
                    onClick={onSelect}
                    className={`flex-1 text-left px-4 py-3 rounded-lg border transition-all duration-160 font-medium text-[15px] relative overflow-hidden ${isSelected
                        ? 'bg-node-active border-black/[0.08] text-foreground'
                        : 'bg-node/40 border-black/[0.03] hover:bg-node/60 text-foreground/70'
                        }`}
                >
                    {/* Selection accent line */}
                    {isSelected && <div className="absolute left-0 top-0 bottom-0 w-[2.5px] bg-blue-500/60" />}
                    {path}
                </button>
            </div>

            {/* Accordion List with Grid Template Rows for smoothness */}
            <div
                className={`grid transition-all duration-200 ease-in-out ${isExpanded ? 'grid-rows-[1fr] opacity-100 mt-1 mb-2' : 'grid-rows-[0fr] opacity-0'}`}
            >
                <div className="overflow-hidden">
                    {children}
                </div>
            </div>

            {/* Spacing if not last */}
            {!isLast && <div className="h-5" />}
            {isLast && <div className="h-2" />}
        </div>
    );
});

FileNode.displayName = 'FileNode';
