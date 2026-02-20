/**
 * ML Automation plugin for OpenCode.ai
 *
 * Injects ML automation bootstrap context via system prompt transform.
 * Skills are discovered via OpenCode's native skill tool from symlinked directory.
 */

import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const extractAndStripFrontmatter = (content) => {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { frontmatter: {}, content };

  const frontmatterStr = match[1];
  const body = match[2];
  const frontmatter = {};

  for (const line of frontmatterStr.split('\n')) {
    const colonIdx = line.indexOf(':');
    if (colonIdx > 0) {
      const key = line.slice(0, colonIdx).trim();
      const value = line.slice(colonIdx + 1).trim().replace(/^["']|["']$/g, '');
      frontmatter[key] = value;
    }
  }

  return { frontmatter, content: body };
};

const MLAutomationPlugin = async ({ project, client, $, directory, worktree }) => {
  const homeDir = os.homedir();
  const commandsDir = path.resolve(__dirname, '../../commands');
  const agentsDir = path.resolve(__dirname, '../../agents');
  const configDir = process.env.OPENCODE_CONFIG_DIR || path.join(homeDir, '.config/opencode');

  // Build lists once at plugin load, not on every transform call
  let skillList = '(Could not list skills)';
  try {
    const files = fs.readdirSync(commandsDir).filter(f => f.endsWith('.md'));
    skillList = files.map(f => {
      const content = fs.readFileSync(path.join(commandsDir, f), 'utf8');
      const { frontmatter } = extractAndStripFrontmatter(content);
      return `- **ml-automation/${frontmatter.name || f.replace('.md', '')}**: ${frontmatter.description || 'No description'}`;
    }).join('\n');
  } catch {}

  let agentList = '(Could not list agents)';
  try {
    const files = fs.readdirSync(agentsDir).filter(f => f.endsWith('.md'));
    agentList = files.map(f => {
      const content = fs.readFileSync(path.join(agentsDir, f), 'utf8');
      const lines = content.split('\n');
      const nameLine = lines.find(l => l.startsWith('# '));
      return `- **${f.replace('.md', '')}**: ${nameLine ? nameLine.replace('# ', '') : f}`;
    }).join('\n');
  } catch {}

  // Pre-build the bootstrap string once
  const bootstrap = `<ML_AUTOMATION_PLUGIN>
You have the ML Automation plugin installed.

## Available Skills (load with skill tool)
${skillList}

## Available Agents
${agentList}

## Tool Mapping for OpenCode
When skills reference Claude Code tools, substitute OpenCode equivalents:
- \`TodoWrite\` → \`update_plan\`
- \`Task\` tool with subagents → Use OpenCode's subagent system (@mention)
- \`Skill\` tool → OpenCode's native \`skill\` tool
- \`Read\`, \`Write\`, \`Edit\`, \`Bash\` → Your native tools

## Skills Location
ML Automation skills are in \`${configDir}/skills/ml-automation/\`
Use OpenCode's native \`skill\` tool to list and load skills.
</ML_AUTOMATION_PLUGIN>`;

  return {
    'experimental.chat.system.transform': async (_input, output) => {
      output.system.push(bootstrap);
    }
  };
};

export default MLAutomationPlugin;
