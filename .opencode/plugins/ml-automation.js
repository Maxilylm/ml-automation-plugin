/**
 * ML Automation plugin for OpenCode.ai
 *
 * Registers ML workflow tools (eda, train, deploy, etc.) and injects
 * bootstrap context into the system prompt. Each tool reads its command
 * markdown and returns it as instructions for the AI to follow.
 */

import path from 'path';
import fs from 'fs';
import os from 'os';
import { tool } from '@opencode-ai/plugin';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const extractAndStripFrontmatter = (content) => {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { frontmatter: {}, body: content };
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
  return { frontmatter, body };
};

const MLAutomationPlugin = async ({ project, client, $, directory, worktree }) => {
  const homeDir = os.homedir();
  const commandsDir = path.resolve(__dirname, '../../commands');
  const agentsDir = path.resolve(__dirname, '../../agents');

  // Cache all command files at load time
  const commands = {};
  try {
    const files = fs.readdirSync(commandsDir).filter(f => f.endsWith('.md'));
    for (const f of files) {
      const raw = fs.readFileSync(path.join(commandsDir, f), 'utf8');
      const { frontmatter, body } = extractAndStripFrontmatter(raw);
      const name = frontmatter.name || f.replace('.md', '');
      commands[name] = { description: frontmatter.description || '', body };
    }
  } catch {}

  // Cache agent list at load time
  let agentList = '';
  try {
    const files = fs.readdirSync(agentsDir).filter(f => f.endsWith('.md'));
    agentList = files.map(f => {
      const content = fs.readFileSync(path.join(agentsDir, f), 'utf8');
      const lines = content.split('\n');
      const nameLine = lines.find(l => l.startsWith('# '));
      return `- **${f.replace('.md', '')}**: ${nameLine ? nameLine.replace('# ', '') : f}`;
    }).join('\n');
  } catch {}

  // Build tools from cached commands
  const tools = {};

  for (const [name, cmd] of Object.entries(commands)) {
    tools[`ml_${name.replace(/-/g, '_')}`] = tool({
      description: cmd.description,
      args: {
        args: tool.schema.string().optional(),
      },
      async execute(input) {
        const userArgs = input.args || '';
        return [
          `<ML_AUTOMATION_COMMAND name="${name}" args="${userArgs}">`,
          cmd.body.trim(),
          userArgs ? `\n## User Arguments\n${userArgs}` : '',
          '</ML_AUTOMATION_COMMAND>',
        ].join('\n');
      },
    });
  }

  // System prompt bootstrap
  const toolNames = Object.keys(tools).map(t => `\`${t}\``).join(', ');
  const bootstrap = `<ML_AUTOMATION_PLUGIN>
You have the ML Automation plugin installed with these tools: ${toolNames}

## Available Agents
${agentList}

## How to Use
- Call any ml_* tool to get workflow instructions, then follow them
- Pass dataset paths, targets, or flags via the \`args\` parameter
- Examples: ml_eda(args: "sales.csv"), ml_train(args: "--target Revenue"), ml_deploy(args: "local")
</ML_AUTOMATION_PLUGIN>`;

  return {
    tool: tools,
    'experimental.chat.system.transform': async (_input, output) => {
      output.system.push(bootstrap);
    },
  };
};

export default MLAutomationPlugin;
