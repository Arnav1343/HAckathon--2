import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: Request) {
    const { idea } = await request.json();

    if (!idea) {
        return NextResponse.json({ error: 'No idea provided' }, { status: 400 });
    }

    // Path to the Python orchestrator
    // Assuming we are in web/src/app/api/project/route.ts
    // Project root is 4 levels up
    const projectRoot = path.resolve(process.cwd(), '..');
    const pythonPath = 'python'; // Or path to your venv python
    const scriptPath = path.join(projectRoot, 'web_orchestrator.py');

    return new Promise((resolve) => {
        const pythonProcess = spawn(pythonPath, [scriptPath, idea], {
            cwd: projectRoot,
            env: {
                ...process.env,
                SPECFORGE_PROVIDER: 'backboard',
                BACKBOARD_API_KEY: 'espr_-xJfRq2-EDhWYk0SvlNpZJjvZ-HJOU9HB5VriVdz1OY',
                SPECFORGE_MODEL: 'gpt-4o-mini',
                PYTHONPATH: projectRoot
            }
        });

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`Python script failed with code ${code}`);
                console.error(`Stderr: ${stderr}`);
                resolve(NextResponse.json({ error: 'Pipeline failed', details: stderr }, { status: 500 }));
                return;
            }

            try {
                // Find the first line that is a valid JSON (in case there are logs)
                const lines = stdout.split('\n');
                const lastLine = lines.filter(l => l.trim()).pop();
                if (!lastLine) throw new Error("No output from script");

                const project = JSON.parse(lastLine);
                resolve(NextResponse.json(project));
            } catch (err) {
                console.error('Failed to parse Python output:', err);
                console.error('Raw stdout:', stdout);
                resolve(NextResponse.json({ error: 'Invalid output from pipeline' }, { status: 500 }));
            }
        });
    });
}
