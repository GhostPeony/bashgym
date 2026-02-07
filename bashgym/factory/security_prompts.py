"""
Security Domain Prompts and Templates

System prompts and message template functions for converting public security
datasets (PhishTank, URLhaus, EMBER, MalwareBazaar, CIC-IDS) into NeMo-compatible
training examples.

Two modes:
- Direct: Template-based responses, no API calls
- Enriched: LLM generates detailed reasoning chains (uses build_enrichment_prompt)
"""

from typing import Any, Dict, Optional


# =============================================================================
# System Prompts (one per security domain)
# =============================================================================

PHISHING_SYSTEM_PROMPT = (
    "You are a phishing detection expert specializing in URL analysis and "
    "social engineering detection. You analyze URLs, email content, and web "
    "artifacts to determine whether they are phishing attempts or legitimate. "
    "You explain your reasoning clearly, citing specific indicators such as "
    "domain impersonation, suspicious TLDs, URL obfuscation, and brand "
    "spoofing patterns."
)

MALWARE_SYSTEM_PROMPT = (
    "You are a malware analysis expert specializing in static and dynamic "
    "analysis of executable files. You analyze PE headers, import tables, "
    "section entropy, and behavioral indicators to classify files as malware "
    "or benign. You explain your reasoning clearly, citing specific indicators "
    "such as packing, obfuscation, suspicious API imports, and anomalous "
    "section characteristics."
)

NETWORK_SYSTEM_PROMPT = (
    "You are a network intrusion detection expert specializing in flow-based "
    "traffic analysis. You analyze network flow features including packet "
    "counts, byte volumes, flow durations, flag distributions, and protocol "
    "characteristics to detect attacks such as DDoS, port scanning, brute "
    "force, and botnet activity. You explain your reasoning clearly, citing "
    "specific flow anomalies."
)


# =============================================================================
# User Prompt Templates
# =============================================================================

def phishing_user_prompt(sample: Dict[str, Any]) -> str:
    """Build user prompt for phishing analysis from a dataset sample."""
    url = sample.get("url", "N/A")
    target = sample.get("target", sample.get("target_brand", ""))
    submission_time = sample.get("submission_time", sample.get("dateadded", ""))

    parts = ["Analyze the following URL for phishing indicators:\n"]
    parts.append(f"URL: {url}")
    if target:
        parts.append(f"Reported target brand: {target}")
    if submission_time:
        parts.append(f"Reported: {submission_time}")

    # URLhaus-specific fields
    threat = sample.get("threat", "")
    if threat:
        parts.append(f"Threat type: {threat}")
    tags = sample.get("tags", "")
    if tags:
        parts.append(f"Tags: {tags}")

    parts.append("\nClassify this URL as phishing or legitimate and explain your reasoning.")
    return "\n".join(parts)


def malware_user_prompt(sample: Dict[str, Any]) -> str:
    """Build user prompt for malware analysis from a dataset sample."""
    parts = ["Analyze the following file for malware indicators:\n"]

    sha256 = sample.get("sha256", sample.get("sha256_hash", ""))
    if sha256:
        parts.append(f"SHA256: {sha256}")

    # EMBER fields
    file_size = sample.get("size", sample.get("file_size", ""))
    if file_size:
        parts.append(f"File size: {file_size} bytes")

    has_debug = sample.get("has_debug", "")
    if has_debug != "":
        parts.append(f"Has debug info: {has_debug}")

    num_imports = sample.get("num_imports", sample.get("import_count", ""))
    if num_imports != "":
        parts.append(f"Import count: {num_imports}")

    num_exports = sample.get("num_exports", sample.get("export_count", ""))
    if num_exports != "":
        parts.append(f"Export count: {num_exports}")

    entropy = sample.get("entropy", sample.get("string_entropy", ""))
    if entropy != "":
        parts.append(f"String entropy: {entropy}")

    num_sections = sample.get("num_sections", "")
    if num_sections != "":
        parts.append(f"Number of sections: {num_sections}")

    # MalwareBazaar fields
    file_type = sample.get("file_type", sample.get("file_type_mime", ""))
    if file_type:
        parts.append(f"File type: {file_type}")

    signature = sample.get("signature", "")
    if signature:
        parts.append(f"Signature: {signature}")

    reporter = sample.get("reporter", "")
    if reporter:
        parts.append(f"Reporter: {reporter}")

    tags = sample.get("tags", "")
    if tags:
        if isinstance(tags, list):
            tags = ", ".join(tags)
        parts.append(f"Tags: {tags}")

    parts.append("\nClassify this file as malware or benign and explain your reasoning.")
    return "\n".join(parts)


def network_user_prompt(sample: Dict[str, Any]) -> str:
    """Build user prompt for network intrusion detection from a CIC-IDS sample."""
    parts = ["Analyze the following network flow for intrusion indicators:\n"]

    # Core flow features (CIC-IDS column names, with whitespace-stripped variants)
    flow_fields = [
        ("Flow Duration", "flow_duration"),
        ("Total Fwd Packets", "total_fwd_packets"),
        ("Total Backward Packets", "total_bwd_packets"),
        ("Total Length of Fwd Packets", "total_length_fwd_packets"),
        ("Total Length of Bwd Packets", "total_length_bwd_packets"),
        ("Flow Bytes/s", "flow_bytes_s"),
        ("Flow Packets/s", "flow_packets_s"),
        ("Fwd Packet Length Mean", "fwd_packet_length_mean"),
        ("Bwd Packet Length Mean", "bwd_packet_length_mean"),
        ("Flow IAT Mean", "flow_iat_mean"),
        ("Fwd IAT Mean", "fwd_iat_mean"),
        ("Bwd IAT Mean", "bwd_iat_mean"),
        ("Fwd PSH Flags", "fwd_psh_flags"),
        ("SYN Flag Count", "syn_flag_count"),
        ("RST Flag Count", "rst_flag_count"),
        ("ACK Flag Count", "ack_flag_count"),
        ("Destination Port", "destination_port"),
        ("Protocol", "protocol"),
    ]

    for display_name, alt_key in flow_fields:
        # Try original column name, stripped version, and snake_case alt
        val = sample.get(display_name)
        if val is None:
            val = sample.get(display_name.strip())
        if val is None:
            val = sample.get(alt_key)
        if val is not None and str(val).strip() != "":
            parts.append(f"{display_name}: {val}")

    # Source/dest info
    src_ip = sample.get("Source IP", sample.get("src_ip", ""))
    dst_ip = sample.get("Destination IP", sample.get("dst_ip", ""))
    if src_ip:
        parts.append(f"Source IP: {src_ip}")
    if dst_ip:
        parts.append(f"Destination IP: {dst_ip}")

    parts.append("\nClassify this network flow as an attack or benign and explain your reasoning.")
    return "\n".join(parts)


# =============================================================================
# Direct-Mode Response Builders
# =============================================================================

def phishing_assistant_response_direct(sample: Dict[str, Any]) -> str:
    """Build a template-based assistant response for phishing classification."""
    # Determine label
    is_phishing = _is_phishing(sample)
    label = "PHISHING" if is_phishing else "LEGITIMATE"
    url = sample.get("url", "N/A")
    target = sample.get("target", sample.get("target_brand", ""))
    threat = sample.get("threat", "")

    parts = [f"**Classification: {label}**\n"]

    if is_phishing:
        if target:
            parts.append(f"**Impersonated Brand:** {target}")

        parts.append("**URL Analysis:**")
        # Extract domain indicators
        domain = _extract_domain(url)
        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        suspicious_tlds = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "club", "icu"}
        if tld.lower() in suspicious_tlds:
            parts.append(f"- .{tld} TLD is commonly associated with phishing campaigns")
        if target and target.lower() in domain.lower() and target.lower() != domain.lower():
            parts.append(f"- Domain impersonates {target} but is not the official domain")
        if len(domain) > 40:
            parts.append("- Unusually long domain name, common in phishing URLs")
        if any(c in url for c in ["@", "//", "%"]):
            parts.append("- URL contains obfuscation characters")
        if threat:
            parts.append(f"- Associated threat: {threat}")
        parts.append(f"\n**Verdict:** This URL is classified as phishing.")
    else:
        parts.append("**URL Analysis:**")
        parts.append("- No phishing indicators detected")
        parts.append("- Domain appears legitimate")
        parts.append(f"\n**Verdict:** This URL is classified as legitimate.")

    return "\n".join(parts)


def malware_assistant_response_direct(sample: Dict[str, Any]) -> str:
    """Build a template-based assistant response for malware classification."""
    is_malware = _is_malware(sample)
    label = "MALWARE" if is_malware else "BENIGN"

    parts = [f"**Classification: {label}**\n"]
    parts.append("**PE Analysis:**")

    has_debug = sample.get("has_debug", "")
    if str(has_debug) == "0":
        parts.append("- No debug information present (common in malware)")

    num_exports = sample.get("num_exports", sample.get("export_count", ""))
    if str(num_exports) == "0":
        parts.append("- No exports (typical of dropper/loader payloads)")

    entropy = sample.get("entropy", sample.get("string_entropy", ""))
    if entropy != "":
        try:
            ent = float(entropy)
            if ent > 6.5:
                parts.append(f"- High string entropy ({ent:.2f}) suggests obfuscation or packing")
            elif ent < 3.0:
                parts.append(f"- Low string entropy ({ent:.2f}) suggests minimal string content")
            else:
                parts.append(f"- Moderate string entropy ({ent:.2f})")
        except (ValueError, TypeError):
            pass

    num_imports = sample.get("num_imports", sample.get("import_count", ""))
    if num_imports != "":
        try:
            imp_count = int(num_imports)
            if imp_count > 100:
                parts.append(f"- High import count ({imp_count}) may indicate complex functionality")
            elif imp_count < 5:
                parts.append(f"- Very low import count ({imp_count}) suggests runtime resolution")
        except (ValueError, TypeError):
            pass

    file_size = sample.get("size", sample.get("file_size", ""))
    if file_size != "":
        try:
            size = int(file_size)
            if size < 10000:
                parts.append(f"- Small file size ({size} bytes) typical of droppers")
            elif size > 10_000_000:
                parts.append(f"- Large file size ({size} bytes) may contain embedded payloads")
        except (ValueError, TypeError):
            pass

    signature = sample.get("signature", "")
    if signature:
        parts.append(f"- Known malware signature: {signature}")

    tags = sample.get("tags", "")
    if tags:
        if isinstance(tags, list):
            tags = ", ".join(tags)
        parts.append(f"- Associated tags: {tags}")

    verdict = "malware" if is_malware else "benign"
    parts.append(f"\n**Verdict:** This file is classified as {verdict}.")
    return "\n".join(parts)


def network_assistant_response_direct(sample: Dict[str, Any]) -> str:
    """Build a template-based assistant response for network flow classification."""
    label_raw = sample.get("Label", sample.get("label", "BENIGN"))
    is_attack = str(label_raw).strip().upper() != "BENIGN"
    label = str(label_raw).strip() if is_attack else "BENIGN"

    parts = [f"**Classification: {'ATTACK' if is_attack else 'BENIGN'}**"]
    if is_attack:
        parts.append(f"**Attack Type:** {label}\n")

    parts.append("**Flow Analysis:**")

    # Duration analysis
    duration = _safe_float(sample, "Flow Duration", "flow_duration")
    if duration is not None:
        if duration < 1000:
            parts.append(f"- Very short flow duration ({duration:.0f} us) may indicate scanning")
        elif duration > 60_000_000:
            parts.append(f"- Long flow duration ({duration/1_000_000:.1f}s) suggests persistent connection")

    # Packet asymmetry
    fwd_pkts = _safe_float(sample, "Total Fwd Packets", "total_fwd_packets")
    bwd_pkts = _safe_float(sample, "Total Backward Packets", "total_bwd_packets")
    if fwd_pkts is not None and bwd_pkts is not None:
        if fwd_pkts > 0 and bwd_pkts == 0:
            parts.append("- No backward packets (one-directional traffic, possible scanning)")
        elif bwd_pkts > 0 and fwd_pkts / max(bwd_pkts, 1) > 10:
            parts.append(f"- High forward/backward packet ratio ({fwd_pkts:.0f}/{bwd_pkts:.0f})")

    # Flag analysis
    syn_count = _safe_float(sample, "SYN Flag Count", "syn_flag_count")
    if syn_count is not None and syn_count > 5:
        parts.append(f"- Elevated SYN flag count ({syn_count:.0f}) may indicate SYN flood")

    rst_count = _safe_float(sample, "RST Flag Count", "rst_flag_count")
    if rst_count is not None and rst_count > 5:
        parts.append(f"- Elevated RST flag count ({rst_count:.0f}) indicates rejected connections")

    # Bytes/s analysis
    flow_bytes = _safe_float(sample, "Flow Bytes/s", "flow_bytes_s")
    if flow_bytes is not None:
        if flow_bytes > 1_000_000:
            parts.append(f"- High throughput ({flow_bytes:.0f} bytes/s) may indicate DDoS or exfiltration")

    if is_attack:
        parts.append(f"\n**Verdict:** This flow is classified as an attack ({label}).")
    else:
        parts.append("\n**Verdict:** This flow is classified as benign traffic.")

    return "\n".join(parts)


# =============================================================================
# Enrichment Prompt Builder (for enriched mode)
# =============================================================================

def build_enrichment_prompt(
    domain: str,
    sample: Dict[str, Any],
    user_prompt: str,
    direct_response: str
) -> str:
    """Build a prompt asking an LLM to generate a detailed reasoning chain.

    Used in enriched mode: the LLM sees the raw sample + direct classification
    and produces a more thorough analysis.
    """
    return (
        f"You are an expert security analyst. Below is a {domain} analysis task, "
        f"the raw sample data, and a basic classification. Generate a detailed, "
        f"expert-level analysis with step-by-step reasoning.\n\n"
        f"## Task\n{user_prompt}\n\n"
        f"## Basic Classification\n{direct_response}\n\n"
        f"## Raw Sample Data\n{_format_sample(sample)}\n\n"
        f"## Instructions\n"
        f"Write a thorough analysis that:\n"
        f"1. Explains your classification reasoning step by step\n"
        f"2. Cites specific indicators from the data\n"
        f"3. Discusses confidence level and potential false positive/negative considerations\n"
        f"4. Suggests follow-up investigation steps if applicable\n\n"
        f"Respond in the same format as the basic classification but with much more detail."
    )


# =============================================================================
# Helpers
# =============================================================================

def _is_phishing(sample: Dict[str, Any]) -> bool:
    """Determine if a sample represents a phishing URL."""
    # PhishTank: verified=true means confirmed phishing
    if sample.get("verified") in (True, "yes", "true", "True"):
        return True
    # URLhaus: any entry in URLhaus is malicious by default
    if sample.get("url_status") or sample.get("threat"):
        return True
    # Explicit label
    label = str(sample.get("label", sample.get("phish_id", ""))).lower()
    if label in ("phishing", "1", "malicious"):
        return True
    if label in ("legitimate", "0", "benign"):
        return False
    # Default: phishing datasets contain mostly phishing
    return True


def _is_malware(sample: Dict[str, Any]) -> bool:
    """Determine if a sample represents malware."""
    label = sample.get("label", sample.get("avclass", ""))
    if isinstance(label, (int, float)):
        return int(label) == 1
    label_str = str(label).lower()
    if label_str in ("1", "malware", "malicious"):
        return True
    if label_str in ("0", "benign", "legitimate", "goodware"):
        return False
    # MalwareBazaar entries are malware by definition
    if sample.get("sha256_hash") and sample.get("reporter"):
        return True
    return True


def _extract_domain(url: str) -> str:
    """Extract domain from a URL string."""
    url = url.strip()
    # Remove protocol
    for prefix in ("https://", "http://", "ftp://"):
        if url.lower().startswith(prefix):
            url = url[len(prefix):]
            break
    # Remove path and query
    domain = url.split("/")[0].split("?")[0].split("#")[0]
    # Remove port
    if ":" in domain:
        domain = domain.split(":")[0]
    return domain


def _safe_float(
    sample: Dict[str, Any],
    key1: str,
    key2: Optional[str] = None
) -> Optional[float]:
    """Safely extract a float value from a sample dict."""
    val = sample.get(key1)
    if val is None and key2:
        val = sample.get(key2)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _format_sample(sample: Dict[str, Any]) -> str:
    """Format a sample dict as readable key-value pairs."""
    lines = []
    for key, val in sample.items():
        if val is not None and str(val).strip() != "":
            lines.append(f"- {key}: {val}")
    return "\n".join(lines) if lines else "(empty sample)"
