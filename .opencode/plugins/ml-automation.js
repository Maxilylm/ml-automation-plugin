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

// Simple frontmatter extraction
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

// Normalize a path: trim whitespace, expand ~, resolve to absolute
const normalizePath = (p, homeDir) => {
  if (!p || typeof p !== 'string') return null;
  let normalized = p.trim();
  if (!normalized) return null;
  if (normalized.startsWith('~/')) {
    normalized = path.join(homeDir, normalized.slice(2));
  } else if (normalized === '~') {
    normalized = homeDir;
  }
  return path.resolve(normalized);
};

export const MLAutomationPlugin = async ({ client, directory }) => {
  const homeDir = os.homedir();
  const commandsDir = path.resolve(__dirname, '../../plugins/ml-automation/commands');
  const agentsDir = path.resolve(__dirname, '../../plugins/ml-automation/agents');
  const envConfigDir = normalizePath(process.env.OPENCODE_CONFIG_DIR, homeDir);
  const configDir = envConfigDir || path.join(homeDir, '.config/opencode');

  // Build list of available skills from commands directory
  const getSkillList = () => {
    try {
      const files = fs.readdirSync(commandsDir).filter(f => f.endsWith('.md'));
      return files.map(f => {
        const content = fs.readFileSync(path.join(commandsDir, f), 'utf8');
        const { frontmatter } = extractAndStripFrontmatter(content);
        return `- **ml-automation/${frontmatter.name || f.replace('.md', '')}**: ${frontmatter.description || 'No description'}`;
      }).join('\n');
    } catch {
      return '(Could not list skills)';
    }
  };

  // Build list of available agents
  const getAgentList = () => {
    try {
      const files = fs.readdirSync(agentsDir).filter(f => f.endsWith('.md'));
      return files.map(f => {
        const content = fs.readFileSync(path.join(agentsDir, f), 'utf8');
        const lines = content.split('\n');
        const nameLine = lines.find(l => l.startsWith('# '));
        return `- **${f.replace('.md', '')}**: ${nameLine ? nameLine.replace('# ', '') : f}`;
      }).join('\n');
    } catch {
      return '(Could not list agents)';
    }
  };

  return {
    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = `<ML_AUTOMATION_PLUGIN>
You have the ML Automation plugin installed.

## Available Skills (load with skill tool)
${getSkillList()}

## Available Agents
${getAgentList()}

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

      (output.system ||= []).push(bootstrap);
    }
  };
};
