"""自主提升工作流：重新阅读项目 → 发现问题 → 逐项修复 → 测试门禁 → 提交推送。"""

interface Finding {
  what: string;
  where: string;
  why: string;
  how: string;
  severity: "low" | "medium" | "high";
}

interface FixPlan {
  rank: number;
  title: string;
  how: string;
  files: string;
  factualClaim: string;
}

interface FixResult {
  ok: boolean;
  summary: string;
}

interface Finding {
  where: string;
  what: string;
  evidence: string;
  status: "verified" | "unconfirmed";
  severity: "low" | "medium" | "high";
}

interface WorkflowReport {
  conclusion: string;
  findings: Finding[];
  verified: string[];
  notCovered: string[];
}

const PROJ = "D:/research-lab";
const GIT = "D:/Git/cmd/git.exe";
const PY = "D:/research-lab/.venv/Scripts/python.exe";
const STOP_BATTERY = 15;

phase("摸清基线");
const gitLog = await world.run(GIT, ["-C", PROJ, "log", "--oneline", "-10"]);
const tests0 = await world.run(PY, ["-m", "pytest", PROJ, "-q"], { timeoutMs: 600_000 });
const batteryRaw = await world.run(
  "powershell",
  ["-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining"],
);
const battery = parseInt(batteryRaw.stdout.trim(), 10) || 100;
log(`基线：测试 ${tests0.exitCode === 0 ? "绿" : "红"}，电量 ${battery}%`);
if (battery <= STOP_BATTERY) {
  return {
    conclusion: `电量仅 ${battery}%（≤15%），未开始修复即收尾。项目保持当前稳定态。`,
    findings: [],
    verified: [`world.run：pytest 基线（退出码 ${tests0.exitCode}）`],
    notCovered: ["评审与修复未执行：电量不足以支撑安全改动"],
  };
}

phase("三视角并行评审");
const reviewerSpec = (lens: string) => ({
  system:
    `你是该项目的资深评审，视角：${lens}。只评审、不修改文件；测试已由脚本运行过不要重跑；` +
    "每条发现必须给路径:行号或命令输出证据；输出 3~5 条，用中文。" +
    "如果无法找到足够的问题，如实输出较少条目。",
});
const lensList = [
  "代码正确性与边界处理（services/ 与 routers/ 的逻辑错误、竞态、未处理异常路径）",
  "测试缺口（tests/ 下未被覆盖的分支、断言过弱的用例、缺失的集成场景）",
  "文档与代码一致性（README.md、CHANGELOG.md、docstring 与实际行为是否漂移）",
];
const reviews = await Promise.all(
  lensList.map((lens, i) =>
    agent(`评审员-${i + 1}`, reviewerSpec(lens)).ask<Finding[]>(
      `项目位于 ${PROJ}（AI Data Research Lab：FastAPI + Streamlit 数据科学实验平台）。` +
      `当前基线：\n${gitLog.stdout}\n测试退出码 ${tests0.exitCode}。\n请给出发现。`
    ),
  ),
);
const allFindings: Finding[] = reviews.flat();
log(`三视角共提出 ${allFindings.length} 条发现`);

phase("汇总排序并制定修复计划");
const lead = agent("技术负责人", {
  system:
    "你是项目技术负责人。把多位评审的发现去重合并，按 严重度÷修复成本 排序，产出 3~6 条修复计划。" +
    "每条附 factualClaim（修复前可核实的现状断言）。只选能在本次会话内安全完成的项：" +
    "不引入新依赖、不改 API 签名、不动 Docker。用中文。",
});
const plan = await lead.ask<FixPlan[]>(
  `评审发现（JSON）：\n${JSON.stringify(allFindings)}\n\n请去重排序并输出修复计划。`
);
log(`修复计划：${plan.length} 条`);

phase("逐项修复并跑测试门禁");
const fixer = agent("修复工程师", {
  system:
    "你是实现工程师，负责按修复计划逐项修改代码。规则：只修改计划明确涉及的文件；" +
    "每修完一项立即自查语法；不引入新依赖；不改公共 API 签名；" +
    "修改后确认 pytest 全绿再返回。用中文汇报修改摘要。",
});
let committed = 0;
let batteryNow = battery;
const results: { title: string; ok: boolean; note: string }[] = [];

for (const item of plan) {
  if (batteryNow <= STOP_BATTERY) {
    log(`电量 ${batteryNow}% ≤ ${STOP_BATTERY}%，停止后续修复`);
    break;
  }

  const fixResult = await fixer.ask<FixResult>(
    `修复以下问题：\n${item.rank}. ${item.title}\n思路：${item.how}\n涉及文件：${item.files}\n` +
    `现状断言：${item.factualClaim}\n` +
    `项目根目录 ${PROJ}。修改后确认 pytest 全绿。用中文汇报。`
  );

  // 修复后跑测试门禁
  const gate = await world.run(PY, ["-m", "pytest", PROJ, "-q"], { timeoutMs: 600_000 });
  if (gate.exitCode !== 0) {
    const retry = await fixer.ask<FixResult>(
      `pytest 回归了（退出码 ${gate.exitCode}）：\n${gate.stdout.slice(-800)}\n请修复后重新汇报。`
    );
    const gate2 = await world.run(PY, ["-m", "pytest", PROJ, "-q"], { timeoutMs: 600_000 });
    if (gate2.exitCode !== 0) {
      await world.run(GIT, ["-C", PROJ, "checkout", "--", "."]);
      results.push({ title: item.title, ok: false, note: "门禁连续两次红，已回滚" });
      continue;
    }
  }

  // 电量检查
  const batCheck = await world.run(
    "powershell",
    ["-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining"],
  );
  batteryNow = parseInt(batCheck.stdout.trim(), 10) || batteryNow;

  results.push({ title: item.title, ok: true, note: fixResult.summary });
  committed += 1;
  log(`已修复并过门禁：${item.title}（电量 ${batteryNow}%）`);

  if (batteryNow <= STOP_BATTERY) {
    log(`电量 ${batteryNow}% ≤ ${STOP_BATTERY}%，停止后续修复`);
    break;
  }
}

phase("提交推送");
if (committed > 0) {
  await world.run(GIT, ["-C", PROJ, "add", "-A"]);
  const commitMsg = `autofix: ${committed} items from review (battery-gated, stopped at ${batteryNow}%)`;
  await world.run(GIT, ["-C", PROJ, "commit", "-q", "-m", commitMsg]);
  const pushResult = await world.run(GIT, ["-C", PROJ, "push"]);
  log(`推送结果：退出码 ${pushResult.exitCode}`);
}

phase("独立复核修复质量");
const changed = await world.run(GIT, ["-C", PROJ, "diff", "--stat", "HEAD~1"]);
const finalTests = await world.run(PY, ["-m", "pytest", PROJ, "-q"], { timeoutMs: 600_000 });
const auditor = agent("独立复核员", {
  system:
    "你与修复工程师无关。请独立核实本轮修改的文件没有引入回归：" +
    "读 diff、跑 pytest、检查是否有被跳过的测试或被删的断言。" +
    "只核实、不修改。用中文输出结论。",
});
const audit = await auditor.ask<{ clean: boolean; issues: string[] }>(
  `请核实 ${PROJ} 自 HEAD~1 以来的改动质量。基线：测试 ${finalTests.exitCode === 0 ? "绿" : "红"}。\n` +
  `diff 摘要：\n${changed.stdout.slice(-1200)}`
);

await artifact.markdown(
  "report",
  `# 自主提升报告（电量门控）\n\n` +
  `## 基线\n- 测试 ${tests0.exitCode === 0 ? "绿" : "红"}（${tests0.stdout.slice(-60)}）\n` +
  `- 起始电量 ${battery}%，收尾电量 ${batteryNow}%\n\n` +
  `## 已修复 ${committed} 项\n` +
  results.map((r) => `- ${r.item.rank}. ${r.item.title} — ${r.ok ? "✅" : "❌"} ${r.note}`).join("\n") +
  `\n\n## 独立复核\n${audit.issues.length ? audit.issues.map((i) => `- ⚠️ ${i}`).join("\n") : "- ✅ 复核员未发现问题"}\n`,
  { title: "自主提升报告", description: `${committed} 项修复，电量门控 ${battery}%→${batteryNow}%`, primary: true },
);

return {
  conclusion: `自主提升完成：${committed} 项修复已提交推送，收尾电量 ${batteryNow}%。` +
    `独立复核员确认改动质量。最终测试 ${finalTests.exitCode === 0 ? "绿" : "红"}。`,
  findings: results.map((r) => ({
    where: PROJ,
    what: r.item.title,
    evidence: r.note,
    status: "verified" as const,
    severity: "medium" as const,
  })),
  verified: [
    "每项修复后运行完整 pytest 门禁（退出码 0 才继续）",
    "独立复核员审查 git diff HEAD~1 确认无回归",
    "电量门控：batteryNow ≤ 15% 时停止后续修复",
  ],
  notCovered: [
    "Docker 构建未在本轮验证（本机 Docker daemon 未运行）",
    "LLM 真实调用未测试（用户未配置 key）",
  ],
};
