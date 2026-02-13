"use client";

import React, { useState, useCallback, useMemo, memo, useEffect, useRef } from 'react';
import { Project, PromptPack } from '@/lib/types';
import { CoreNode } from './CoreNode';
import { FileNode, FunctionNode } from './FileNode';
import { PromptViewer } from './PromptViewer';

interface ExplorerLayoutProps {
    project: Project;
}

export const ExplorerLayout = memo(({ project }: ExplorerLayoutProps) => {
    const [isProjectExpanded, setIsProjectExpanded] = useState(true);
    const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [localPromptPack, setLocalPromptPack] = useState<PromptPack>(project.prompt_pack);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const selectedRef = useRef<HTMLDivElement>(null);

    const toggleProject = useCallback(() => {
        setIsProjectExpanded(prev => !prev);
    }, []);

    const toggleFile = useCallback((path: string) => (e: React.MouseEvent) => {
        e.stopPropagation();
        setExpandedFiles(prev => {
            const next = new Set(prev);
            if (next.has(path)) {
                next.delete(path);
            } else {
                next.add(path);
            }
            return next;
        });
    }, []);

    const selectFile = useCallback((path: string) => () => {
        setSelectedFile(path);
    }, []);

    // Auto-scroll centering
    useEffect(() => {
        if (selectedFile && selectedRef.current) {
            selectedRef.current.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [selectedFile]);

    const handleEditPrompt = useCallback((path: string, newContent: string) => {
        setLocalPromptPack(prev => prev.map(item =>
            item.path === path ? { ...item, implementation_prompt: newContent } : item
        ));
    }, []);

    const promptMap = useMemo(() => {
        return localPromptPack.reduce((acc, item) => {
            acc[item.path] = item.implementation_prompt;
            return acc;
        }, {} as Record<string, string>);
    }, [localPromptPack]);

    const selectedPrompt = selectedFile ? promptMap[selectedFile] : null;

    return (
        <div className="flex h-screen w-full overflow-hidden bg-background">
            {/* Explorer Column (Fixed 45%) */}
            <div
                ref={scrollContainerRef}
                className="w-[45%] h-full overflow-y-auto border-r border-black/[0.04] scroll-smooth"
            >
                <div className="max-w-2xl mx-auto py-16 px-8">
                    <CoreNode
                        title={project.spec.project_name}
                        isExpanded={isProjectExpanded}
                        onToggle={toggleProject}
                    />

                    {isProjectExpanded && (
                        <div className="flex flex-col mt-4">
                            {project.architecture.files.map((file, idx) => {
                                const isLastFile = idx === project.architecture.files.length - 1;
                                const isSelected = selectedFile === file.path;
                                const functions = project.contracts[file.path]?.functions || [];

                                return (
                                    <div key={file.path} ref={isSelected ? selectedRef : null}>
                                        <FileNode
                                            path={file.path}
                                            isExpanded={expandedFiles.has(file.path)}
                                            isSelected={isSelected}
                                            onToggle={toggleFile(file.path)}
                                            onSelect={selectFile(file.path)}
                                            isLast={isLastFile}
                                        >
                                            {functions.map((fn, fIdx) => (
                                                <FunctionNode
                                                    key={`${file.path}-${fn.name}-${fIdx}`}
                                                    contract={fn}
                                                    isLast={fIdx === functions.length - 1}
                                                />
                                            ))}
                                        </FileNode>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>

            {/* Prompt View (Fixed 55%) */}
            <div className="w-[55%] h-full flex-shrink-0 bg-panel">
                <PromptViewer
                    content={selectedPrompt}
                    fileName={selectedFile}
                    projectName={project.spec.project_name}
                    onEdit={selectedFile ? (val: string) => handleEditPrompt(selectedFile, val) : undefined}
                />
            </div>
        </div>
    );
});

ExplorerLayout.displayName = 'ExplorerLayout';
