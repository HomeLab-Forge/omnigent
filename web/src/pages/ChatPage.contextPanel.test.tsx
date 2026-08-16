import type * as UseWorkspaceChangedFilesModule from "@/hooks/useWorkspaceChangedFiles";
import type * as UseSessionModule from "@/hooks/useSession";
import type * as UseHostsModule from "@/hooks/useHosts";
import type * as RunnerHealthProviderModule from "@/hooks/RunnerHealthProvider";
import type * as AgentLabelsModule from "@/lib/agentLabels";

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useChatStore } from "@/store/chatStore";

// Composer reads workspace files via a TanStack query hook (for "@"-file
// mentions); these tests don't exercise it, so stub the hook to avoid wrapping
// every render in a QueryClientProvider.
vi.mock("@/hooks/useWorkspaceChangedFiles", async (importOriginal) => {
  const actual = await importOriginal<typeof UseWorkspaceChangedFilesModule>();
  return {
    ...actual,
    useWorkspaceAllFiles: () => ({ data: undefined }),
    useWorkspaceDirectory: () => ({ data: undefined }),
  };
});
// HostBadge renders in the composer's status-line tray and reads the session's
// host binding through these hooks. Stub them so it self-hides (no host bound)
// without needing a QueryClient / RunnerHealth provider.
vi.mock("@/hooks/useSession", async (importOriginal) => ({
  ...(await importOriginal<typeof UseSessionModule>()),
  useSession: () => ({ session: { hostId: null }, isLoading: false, error: null }),
}));
vi.mock("@/hooks/useHosts", async (importOriginal) => ({
  ...(await importOriginal<typeof UseHostsModule>()),
  useHosts: () => ({ data: [] }),
}));
vi.mock("@/hooks/RunnerHealthProvider", async (importOriginal) => ({
  ...(await importOriginal<typeof RunnerHealthProviderModule>()),
  useSessionHostOnline: () => undefined,
}));
vi.mock("@/lib/agentLabels", async (importOriginal) => ({
  ...(await importOriginal<typeof AgentLabelsModule>()),
  useBrainHarnessLabels: () => ({
    "claude-sdk": "Claude SDK",
    codex: "Codex",
    cursor: "Cursor",
    pi: "Pi",
    antigravity: "Antigravity",
    copilot: "Copilot",
  }),
}));

import { Composer } from "./ChatPage";

// Pins "/context" as a toggle over a live panel rather than a one-shot text
// snapshot. The snapshot was written into the composer's inline command
// feedback, which the next keystroke cleared and which subscribed to nothing —
// so it vanished as soon as the user started typing and never moved while a
// turn ran.

/** Minimal ComposerProps for an interactive (writable, idle) composer. */
function composerProps(overrides: Partial<Parameters<typeof Composer>[0]> = {}) {
  return {
    status: "idle" as const,
    isWorking: false,
    disabled: false,
    onSend: vi.fn(),
    onStop: vi.fn(),
    agents: undefined,
    selectedAgentId: null,
    permissionLevel: null,
    readOnlyReason: null,
    replyQuotes: [],
    onRemoveQuote: vi.fn(),
    onClearAllQuotes: vi.fn(),
    effortLevels: ["low", "medium", "high"] as const,
    showEffort: true,
    showModels: false,
    modelPickerKind: null,
    codexModelOptions: [],
    showCodexPlanMode: false,
    ...overrides,
  };
}

function renderComposer(overrides: Partial<Parameters<typeof Composer>[0]> = {}) {
  return render(
    <TooltipProvider>
      <Composer {...composerProps(overrides)} />
    </TooltipProvider>,
  );
}

/** The composer textarea, located by its aria-label. */
function textarea(): HTMLTextAreaElement {
  return screen.getByLabelText("Message the agent") as HTMLTextAreaElement;
}

/** The context panel, or null while it is closed. */
function panel(): HTMLElement | null {
  return screen.queryByTestId("composer-context-panel");
}

/**
 * Submit `text` from the composer. The trailing space keeps the suggestions
 * menu closed (it opens only while the command name is still being typed), so
 * Enter reaches ``submit()`` — the path the toggle has to survive.
 */
function submitText(text: string) {
  const ta = textarea();
  fireEvent.change(ta, { target: { value: text } });
  fireEvent.keyDown(ta, { key: "Enter" });
}

describe("Composer /context panel", () => {
  beforeEach(() => {
    useChatStore.setState({
      conversationId: "conv_test",
      skills: [],
      blocks: [],
      contextWindow: 100_000,
      tokensUsed: 25_000,
      llmModel: "claude-opus-5",
      sessionModelOverride: null,
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("stays closed until /context runs", () => {
    renderComposer();
    expect(panel()).toBeNull();
  });

  it("opens on /context with the model, usage, and item count", () => {
    renderComposer();
    submitText("/context ");

    const open = panel();
    expect(open).not.toBeNull();
    expect(open).toHaveTextContent("claude-opus-5");
    expect(screen.getByTestId("composer-context-tokens")).toHaveTextContent("25.0%");
    expect(screen.getByRole("progressbar", { name: "Context used" })).toHaveAttribute(
      "aria-valuenow",
      "25",
    );
    expect(open).toHaveTextContent("Items in context: 0");
  });

  it("names the session override ahead of the bound model", () => {
    useChatStore.setState({ sessionModelOverride: "claude-sonnet-5" });
    renderComposer();
    submitText("/context ");
    expect(panel()).toHaveTextContent("claude-sonnet-5 (override)");
  });

  it("survives the keystrokes of the next message", () => {
    renderComposer();
    submitText("/context ");
    expect(panel()).not.toBeNull();

    // The old snapshot lived in commandError, which onChange clears.
    fireEvent.change(textarea(), { target: { value: "what is left?" } });
    expect(panel()).not.toBeNull();
  });

  it("tracks usage that arrives after it opened", () => {
    renderComposer();
    submitText("/context ");
    expect(screen.getByTestId("composer-context-tokens")).toHaveTextContent("25.0%");

    act(() => {
      useChatStore.setState({ tokensUsed: 80_000 });
    });
    expect(screen.getByTestId("composer-context-tokens")).toHaveTextContent("80.0%");
  });

  it("closes on a second /context", () => {
    // The toggle's functional update is why closing can't happen at the top of
    // submit(): "/context" returns through the slash-command branch, so a close
    // batched with the toggle reads the pending false and flips it back open.
    renderComposer();
    submitText("/context ");
    expect(panel()).not.toBeNull();

    submitText("/context ");
    expect(panel()).toBeNull();
  });

  it("closes when a message is sent", () => {
    const onSend = vi.fn();
    renderComposer({ onSend });
    submitText("/context ");
    expect(panel()).not.toBeNull();

    submitText("how much room is left?");
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(panel()).toBeNull();
  });

  it("closes from its own dismiss button", () => {
    renderComposer();
    submitText("/context ");
    fireEvent.click(screen.getByLabelText("Close context usage"));
    expect(panel()).toBeNull();
  });

  it("says so when no usage has arrived yet", () => {
    useChatStore.setState({ tokensUsed: null, contextWindow: null });
    renderComposer();
    submitText("/context ");
    expect(screen.getByTestId("composer-context-tokens")).toHaveTextContent("No usage data yet");
    expect(screen.queryByRole("progressbar")).toBeNull();
  });
});
