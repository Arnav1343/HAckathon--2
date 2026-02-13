import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(
    request: Request,
    { params }: { params: { id: string } }
) {
    const id = (await params).id;
    const projectRoot = path.resolve(process.cwd(), '..');
    const projectPath = path.join(projectRoot, 'builds', 'web_projects', `${id}.json`);

    try {
        if (fs.existsSync(projectPath)) {
            const data = fs.readFileSync(projectPath, 'utf8');
            return NextResponse.json(JSON.parse(data));
        } else {
            return NextResponse.json({ error: 'Project not found' }, { status: 404 });
        }
    } catch (err) {
        console.error('Failed to read project:', err);
        return NextResponse.json({ error: 'Failed to load project' }, { status: 500 });
    }
}
