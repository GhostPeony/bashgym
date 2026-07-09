// Strip ANSI escape sequences from raw PTY data so regex patterns can match cleanly
// eslint-disable-next-line no-control-regex
export const ANSI_RE = /[\x1b\x9b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]/g

export function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, '')
}
