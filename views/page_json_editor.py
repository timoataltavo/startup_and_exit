from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
import streamlit as st


def _validate_structure(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (level, message) tuples. level in {INFO, WARN, ERROR}."""
    msgs: List[Tuple[str, str]] = []
    if not isinstance(data, dict):
        return [("ERROR", "Root ist kein Objekt (dict)")]
    # Basic required keys
    if "events" not in data or not isinstance(data.get("events"), list):
        msgs.append(("ERROR", "'events' fehlt oder ist nicht eine Liste"))
    if "liquidation_terms" in data and not isinstance(data.get("liquidation_terms"), dict):
        msgs.append(("ERROR", "'liquidation_terms' sollte ein Objekt sein"))
    events = data.get("events") if isinstance(data.get("events"), list) else []
    seen_round_names: set[str] = set()
    last_date = "0000-00-00"
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            msgs.append(("ERROR", f"Event #{idx+1} ist kein Objekt")); continue
        kind = ev.get("kind", "investment_round")
        name = str(ev.get("name", "")).strip()
        date = ev.get("date")
        if not name:
            msgs.append(("ERROR", f"Event #{idx+1} hat keinen Namen"))
        if kind not in {"investment_round", "vsp_issue"}:
            msgs.append(("WARN", f"Event '{name or idx+1}' unbekannter kind='{kind}'"))
        if date and not _looks_like_date(date):
            msgs.append(("ERROR", f"Event '{name}' Datum nicht im Format YYYY-MM-DD"))
        if date and _looks_like_date(date) and date < last_date:
            msgs.append(("WARN", f"Event '{name}' Datum früher als vorheriges – chronologische Reihenfolge prüfen"))
        last_date = date or last_date
        if kind == "investment_round":
            am = ev.get("amounts_invested", {}) or {}
            sr = ev.get("shares_received", {}) or {}
            if not isinstance(am, dict):
                msgs.append(("ERROR", f"Event '{name}' amounts_invested muss Objekt sein"))
            if not isinstance(sr, dict):
                msgs.append(("ERROR", f"Event '{name}' shares_received muss Objekt sein"))
            if isinstance(am, dict) and sum(_coerce_float(v) for v in am.values()) <= 0:
                msgs.append(("ERROR", f"Event '{name}' keine Investitionsbeträge (>0)"))
            if isinstance(sr, dict) and sum(_coerce_float(v) for v in sr.values()) <= 0:
                msgs.append(("ERROR", f"Event '{name}' keine neuen Anteile (>0)"))
            if name in seen_round_names:
                msgs.append(("ERROR", f"Rundenname '{name}' doppelt"))
            seen_round_names.add(name)
            stv = ev.get("shares_to_vsp")
            if stv is not None and not isinstance(stv, dict):
                msgs.append(("ERROR", f"Event '{name}' shares_to_vsp muss Objekt sein (oder weglassen)"))
        elif kind == "vsp_issue":
            gi = ev.get("vsp_received", {}) or {}
            if not isinstance(gi, dict):
                msgs.append(("ERROR", f"Event '{name}' vsp_received muss Objekt sein"))
            if isinstance(gi, dict) and not gi:
                msgs.append(("WARN", f"Event '{name}' hat leere vsp_received"))
    # liquidation classes cross-check
    liq = data.get("liquidation_terms", {}) if isinstance(data.get("liquidation_terms"), dict) else {}
    classes = liq.get("classes", []) if isinstance(liq.get("classes"), list) else []
    round_names = {e.get("name") for e in events if isinstance(e, dict)}
    for c in classes:
        if not isinstance(c, dict):
            msgs.append(("ERROR", "Eintrag in liquidation_terms.classes ist kein Objekt")); continue
        cname = c.get("name")
        if not cname:
            msgs.append(("ERROR", "LP-Klasse ohne Namen"))
        ap = c.get("applies_to_round_names", []) or []
        if not isinstance(ap, list):
            msgs.append(("ERROR", f"LP-Klasse '{cname}' applies_to_round_names muss Liste sein"))
        else:
            for rn in ap:
                if rn not in round_names:
                    msgs.append(("ERROR", f"LP-Klasse '{cname}' referenziert unbekannte Runde '{rn}'"))
        rate = c.get("simple_interest_rate")
        if rate is not None and not _is_number(rate):
            msgs.append(("ERROR", f"LP-Klasse '{cname}' simple_interest_rate muss Zahl sein"))
        cap = c.get("cap_multiple_total")
        if cap is not None and not _is_number(cap):
            msgs.append(("ERROR", f"LP-Klasse '{cname}' cap_multiple_total muss Zahl oder null sein"))
    if not any(lvl == "ERROR" for lvl, _ in msgs):
        msgs.append(("INFO", "Basis-Struktur ohne kritische Fehler."))
    return msgs


def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_date(s: Any) -> bool:
    if not isinstance(s, str) or len(s) != 10:
        return False
    y, m, d = s.split('-') if '-' in s else (None, None, None)
    return bool(y and m and d and len(y) == 4 and len(m) == 2 and len(d) == 2 and y.isdigit() and m.isdigit() and d.isdigit())


def render(raw_data: Dict[str, Any]):
    st.title("🛠️ JSON Editor")
    st.markdown("Bestehende Daten bearbeiten. Änderungen beeinflussen erst die App nach Übernahme. Vor Export wird validiert.")

    if "json_editor_text" not in st.session_state:
        st.session_state.json_editor_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
    if "show_json_line_numbers" not in st.session_state:
        st.session_state.show_json_line_numbers = False

    st.text_area("JSON bearbeiten", key="json_editor_text", height=500)

    # Optional line-numbered preview (helps when errors reference lines)
    with st.expander("Zeilennummern Vorschau", expanded=st.session_state.show_json_line_numbers):
        try:
            # st.code supports line_numbers in recent Streamlit versions; fallback otherwise
            st.code(st.session_state.json_editor_text, language="json", line_numbers=True)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - older Streamlit fallback
            st.code(st.session_state.json_editor_text, language="json")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Validieren", type="secondary"):
            st.session_state.json_editor_validation_trigger = True
    with col2:
        apply_clicked = st.button("Übernehmen", type="primary")
    with col3:
        export_clicked = st.button("Exportieren (Download)")

    validation_messages: List[Tuple[str,str]] = []
    parsed: Dict[str, Any] | None = None
    if st.session_state.get("json_editor_validation_trigger") or apply_clicked or export_clicked:
        try:
            parsed = json.loads(st.session_state.json_editor_text)
            if not isinstance(parsed, dict):
                st.error("Root muss ein JSON Objekt sein.")
            else:
                validation_messages = _validate_structure(parsed)
        except json.JSONDecodeError as e:
            st.error(f"JSON Syntaxfehler: {e}")
            # Auto-open line numbers to help user locate the issue
            st.session_state.show_json_line_numbers = True

    if validation_messages:
        lvl_to_fn = {"ERROR": st.error, "WARN": st.warning, "INFO": st.info}
        error_count = 0
        for lvl, msg in validation_messages:
            if lvl == "ERROR":
                error_count += 1
            lvl_to_fn.get(lvl, st.write)(f"[{lvl}] {msg}")
        if parsed is not None and apply_clicked and error_count == 0:
            st.session_state.data_store = parsed
            st.success("Neue JSON übernommen (Session aktualisiert).")
        elif apply_clicked and error_count > 0:
            st.warning("Übernahme blockiert – zuerst Fehler beheben.")
        if parsed is not None and export_clicked and error_count == 0:
            out = json.dumps(parsed, ensure_ascii=False, indent=2)
            st.download_button("Download bereit", data=out, file_name="cap_table_custom.json", mime="application/json")
        elif export_clicked and error_count > 0:
            st.warning("Export blockiert – zuerst Fehler beheben.")
    else:
        st.caption("Noch keine Validierung durchgeführt.")
