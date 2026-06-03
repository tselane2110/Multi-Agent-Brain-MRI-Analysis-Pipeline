# agents/mri_agents.py
# -------------------------------------------------------
# This file defines all 7 agents in our pipeline.
#
# Each agent is just a Python function that:
#   1. Receives the current STATE
#   2. Does its specific job (calls the LLM with a focused prompt)
#   3. Returns UPDATES to the state (only the fields it fills in)
#
# LangGraph will call these functions in order and merge
# the returned updates into the shared state automatically.
# -------------------------------------------------------

from langchain_core.messages import HumanMessage
from state import MRIAnalysisState
from llm_setup import get_vision_llm, get_text_llm


# ═══════════════════════════════════════════════════════
# AGENT 0: GATEKEEPER AGENT
# Job: Runs FIRST before anything else.
# Looks at the image and decides: is this actually a brain MRI?
# If not → sets is_brain_mri = False and the pipeline stops.
# If yes → pipeline continues normally.
#
# This is the "input validation" layer of the agentic system.
# In production AI systems, you always validate inputs before
# expensive downstream processing.
# ═══════════════════════════════════════════════════════

def gatekeeper_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 0: Input Validation / Gatekeeper

    Uses the vision LLM to verify the uploaded image is actually
    a brain MRI before allowing the pipeline to proceed.
    Rejects: non-medical images, non-brain scans (chest X-ray, knee MRI, etc.),
             CT scans mistaken for MRI, photographs, etc.
    """
    print("🛡️  [Agent 1/7] Gatekeeper Agent running...")

    vision_llm = get_vision_llm()

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{state['input_image_b64']}"
                }
            },
            {
                "type": "text",
                "text": """You are a medical imaging classifier. Your ONLY job is to determine whether the given image is a brain MRI scan.

Carefully examine the image and respond in this EXACT format — nothing else:

VERDICT: [BRAIN_MRI or NOT_BRAIN_MRI]
REASON: [one concise sentence explaining your decision]
CONFIDENCE: [HIGH or MEDIUM or LOW]

Classification rules:
- BRAIN_MRI: Any MRI sequence (T1, T2, FLAIR, DWI, etc.) showing the brain/head, regardless of plane (axial, coronal, sagittal). Partial brain coverage is acceptable.
- NOT_BRAIN_MRI: Photographs, CT scans, X-rays, ultrasounds, non-brain MRIs (spine, knee, abdomen), drawings, screenshots, or any non-medical image.

Be strict. When in doubt, classify as NOT_BRAIN_MRI."""
            }
        ]
    )

    try:
        response = vision_llm.invoke([message])
        text = response.content.strip() # removes leading+trailing whitespace/newlines

        # Parse the structured response
        is_brain_mri = "VERDICT: BRAIN_MRI" in text.upper()

        # Extract the REASON line
        reason = "No reason provided."
        for line in text.splitlines():
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
                break

        if is_brain_mri:
            print(f"   ✅ Valid brain MRI detected. Proceeding.")
        else:
            print(f"   ❌ Not a brain MRI. Pipeline halted. Reason: {reason}")

        return {
            "is_brain_mri": is_brain_mri,
            "gatekeeper_reason": reason,
        }

    except Exception as e:
        # If gatekeeper itself fails, fail safe — don't proceed
        return {
            "is_brain_mri": False,
            "gatekeeper_reason": f"Gatekeeper agent encountered an error: {str(e)}",
            "error": f"Gatekeeper Agent failed: {str(e)}",
        }


# ═══════════════════════════════════════════════════════
# AGENT 1: PREPROCESSOR AGENT
# Job: Look at the raw MRI image and describe what it sees.
# This gives downstream agents a text description to work with,
# in addition to the raw image. "Grounding" the image in words.
# ═══════════════════════════════════════════════════════

def preprocessor_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 1: Visual Grounding
    
    Uses vision LLM to produce a plain-language description of the MRI.
    Think of this as the agent saying: "Let me look at this scan first
    and tell everyone what I see before we start analyzing."
    """
    print("🔍 [Agent 2/7] Preprocessor Agent running...")
    
    vision_llm = get_vision_llm()
    
    # We send the image + a focused prompt to the vision model
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{state['input_image_b64']}"
                }
            },
            {
                "type": "text",
                "text": """You are a neuroradiology imaging technologist performing pre-read quality assessment.

Examine this brain MRI image systematically and report the following:

1. SEQUENCE IDENTIFICATION
   - Based on signal intensities, identify the likely MRI sequence:
     * T1-weighted: CSF appears dark, white matter brighter than gray matter
     * T2-weighted: CSF appears bright/white, lesions often hyperintense
     * FLAIR (Fluid Attenuated Inversion Recovery): CSF suppressed/dark, periventricular lesions visible
     * DWI (Diffusion Weighted): bright signal = restricted diffusion (acute ischemia)
     * T1+contrast: enhancing lesions appear bright if blood-brain barrier is disrupted
   - State your reasoning for the sequence identification.

2. IMAGING PLANE
   - Identify: axial (horizontal), coronal (front-to-back), or sagittal (side view)
   - Note the approximate slice level if axial (e.g., at the level of basal ganglia, ventricles, vertex)

3. ANATOMICAL INVENTORY
   List which structures are clearly visible in this slice:
   - Cortex and sulci/gyri
   - White matter (centrum semiovale, corona radiata, internal capsule)
   - Basal ganglia (caudate, putamen, globus pallidus)
   - Thalami
   - Ventricles (lateral, 3rd, 4th)
   - Corpus callosum
   - Cerebellum / brainstem (if in field of view)
   - Hippocampi (if in field of view)
   - Meninges / extra-axial spaces

4. IMAGE QUALITY ASSESSMENT
   Rate each: GOOD / ADEQUATE / POOR
   - Signal-to-noise ratio (SNR)
   - Motion artifacts (ghosting, blurring)
   - Susceptibility artifacts (metal, air-tissue interfaces)
   - Slice thickness adequacy
   - Coverage completeness

5. IMMEDIATE VISUAL FLAGS
   Without interpreting clinically, note anything that visually stands out:
   - Obvious signal asymmetries between hemispheres
   - Focal bright or dark spots not expected for this sequence
   - Visible mass effect or structural displacement
   - Obvious sulcal/gyral abnormalities

Be precise and objective. Use standard radiological descriptors (hyperintense, hypointense, isointense, homogeneous, heterogeneous). Do not render diagnoses."""
            }
        ]
    )
    
    try:
        response = vision_llm.invoke([message])
        description = response.content
        
        return {
            "image_description": description,
        }
    except Exception as e:
        return {"error": f"Preprocessor Agent failed: {str(e)}"}

# ═══════════════════════════════════════════════════════
# AGENT 2: ANALYSIS AGENT
# Job: Deep dive into the MRI — find anomalies, measure things,
# flag regions of concern. More focused and technical than Agent 1.
# ═══════════════════════════════════════════════════════

def analysis_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 2: Clinical Analysis
    
    Takes the preprocessor's description + the raw image,
    and performs a systematic clinical analysis looking for
    potential pathological findings.
    """
    print("🧠 [Agent 3/7] Analysis Agent running...")
    
    vision_llm = get_vision_llm()
    
    # Build context from previous agent
    prior_description = state.get("image_description", "No prior description available.")
    patient_context = state.get("patient_context", "No patient context provided.")
    
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{state['input_image_b64']}"
                }
            },
            {
                "type": "text",
                "text": f"""You are an attending neuroradiologist performing a systematic read of a brain MRI.

SEQUENCE & PLANE CONTEXT (from pre-read assessment):
{prior_description}

CLINICAL INDICATION:
{patient_context}

Perform a complete systematic analysis following the ACR (American College of Radiology) structured reporting framework. For each category, describe findings using standard radiological language. Rate each finding's confidence as [HIGH], [MEDIUM], or [LOW].

━━━ SYSTEMATIC ANALYSIS ━━━

1. EXTRA-AXIAL SPACES & CALVARIUM
   - Subdural/epidural spaces: normal vs. collections
   - Subarachnoid spaces: sulcal effacement or widening
   - Skull/calvarium: any focal lesions or erosions
   - Dural sinuses (if visible)

2. CEREBRAL CORTEX & SULCI
   - Gyral pattern: normal vs. simplified/pachygyria vs. polymicrogyria
   - Sulcal depth and symmetry
   - Cortical signal: any focal abnormality, cortical spreading depression signs
   - Signs of cortical atrophy (age-appropriate vs. abnormal)

3. WHITE MATTER
   - Deep white matter: any T2/FLAIR hyperintensities? Distribution pattern?
     (periventricular = MS pattern; subcortical = small vessel disease; juxtacortical = demyelination)
   - Subcortical U-fibers: involved or spared? (sparing = small vessel disease; involvement = leukodystrophy)
   - Internal capsule, corpus callosum: signal and structural integrity
   - Fazekas scale estimate if white matter lesions present (Grade 0-3)

4. DEEP GRAY MATTER (BASAL GANGLIA & THALAMI)
   - Caudate heads: size, signal, symmetry
   - Putamen & globus pallidus: any iron deposition (hypointense on T2), signal abnormality
   - Thalami: size, signal, any focal lesions
   - Signal characteristics relative to expected for sequence

5. VENTRICULAR SYSTEM
   - Lateral ventricles: size (Evans index if measurable), symmetry, wall irregularity
   - Third ventricle: width (>7mm = abnormal), floor depression
   - Fourth ventricle: position, size, fastigial point
   - Aqueduct of Sylvius: patent vs. obstructed
   - Transependymal edema: periventricular FLAIR hyperintensity suggesting raised ICP

6. MIDLINE STRUCTURES
   - Midline shift: present/absent, direction and estimated mm if present
   - Septum pellucidum: intact, cavum, or absent
   - Corpus callosum: complete, partially absent, thin, or dysgenesis

7. POSTERIOR FOSSA
   - Cerebellum: hemispheric symmetry, folia pattern, vermis integrity
   - Brainstem (midbrain, pons, medulla): size, signal, any focal lesions
   - Cerebellar tonsils: position relative to foramen magnum (Chiari malformation check)
   - Cisterns: prepontine, cerebellopontine angle, foramen magnum

8. VASCULAR TERRITORY ASSESSMENT
   - MCA territory (large frontoparietal region): any signal abnormality
   - ACA territory (medial frontal/parietal): involvement?
   - PCA territory (occipital, medial temporal): any ischemic change
   - Watershed zones (between MCA/ACA, MCA/PCA): linear cortical lesions?
   - Flow voids: visible major vessels — MCA, basilar, ICA if visible

9. FOCAL LESIONS (if any identified)
   For EACH lesion describe:
   - Location (lobe, hemisphere, cortical vs. subcortical vs. deep)
   - Size (estimate in mm)
   - Signal on this sequence (hyper/hypo/iso-intense)
   - Margins (sharp/well-defined vs. ill-defined/infiltrative)
   - Mass effect: local vs. regional vs. global
   - Surrounding edema: present/absent, extent
   - Pattern: ring-enhancing? Solid? Cystic? Calcified?

10. OVERALL PATTERN RECOGNITION
    Does the overall pattern suggest any of the following categories?
    - Normal study
    - Focal ischemic/vascular disease
    - Demyelinating disease (MS pattern, ADEM)
    - Neoplastic process (primary vs. metastatic features)
    - Infectious/inflammatory (abscess, encephalitis)
    - Neurodegenerative (atrophy pattern, white matter signal)
    - Traumatic (hemorrhage, DAI pattern)
    - Metabolic/toxic (symmetric deep gray matter involvement)
    - Congenital/developmental variant

For any finding, if confidence is [LOW], explicitly state what additional sequences or clinical information would help confirm it."""
            }
        ]
    )
    
    try:
        response = vision_llm.invoke([message])
        findings = response.content
        
        # Simple extraction of regions of concern
        # (in a production system, you'd parse this more carefully)
        regions_of_concern = _extract_regions_of_concern(findings)
        
        return {
            "findings": findings,
            "regions_of_concern": regions_of_concern,
        }
    except Exception as e:
        return {"error": f"Analysis Agent failed: {str(e)}"}


def _extract_regions_of_concern(findings_text: str) -> str:
    """Helper: extracts a short summary of flagged regions from findings."""
    # Look for lines that aren't "No abnormality detected"
    lines = findings_text.split('\n')
    concerns = []
    for line in lines:
        if line.strip() and "no abnormality" not in line.lower() and len(line) > 20:
            if any(keyword in line.lower() for keyword in 
                   ['hyperintens', 'lesion', 'atrophy', 'shift', 'mass', 'abnormal', 
                    'asymmetr', 'enlarg', 'hemorrh', 'infarct']):
                concerns.append(line.strip())
    
    return "\n".join(concerns) if concerns else "No specific regions of concern identified."


# ═══════════════════════════════════════════════════════
# AGENT 3: REASONING AGENT
# Job: Think step-by-step about what the findings mean.
# This is the "chain of thought" agent — it connects dots.
# "If we see X and Y together, that pattern suggests Z."
# ═══════════════════════════════════════════════════════

def reasoning_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 3: Medical Reasoning
    
    Takes all prior findings and reasons about them like a 
    senior radiologist would — connecting patterns, 
    building differential diagnoses, assessing severity.
    
    This is text-only (no image needed anymore), so we can use Groq/Llama.
    """
    print("💭 [Agent 4/7] Reasoning Agent running...")
    
    if state.get("error"):
        return {}
    
    text_llm = get_text_llm()
    
    # Compile all context gathered so far
    context = f"""
BRAIN MRI ANALYSIS PIPELINE - REASONING STAGE
==============================================

IMAGE DESCRIPTION (from Preprocessor Agent):
{state.get('image_description', 'Not available')}

CLINICAL FINDINGS (from Analysis Agent):
{state.get('findings', 'Not available')}

REGIONS OF CONCERN:
{state.get('regions_of_concern', 'None identified')}

PATIENT CONTEXT:
{state.get('patient_context', 'None provided')}
"""
    
    prompt = f"""{context}

You are a senior neuroradiology consultant performing the clinical reasoning step — the cognitive core of radiology.
Your job is to transform raw observations into clinical meaning using structured radiological reasoning.

━━━ STEP 1: PATTERN SYNTHESIS ━━━
Review all findings from the analysis agent. Group them into:
- PRIMARY FINDING: The most significant or unusual observation
- SECONDARY FINDINGS: Supporting or incidental observations  
- NORMAL FINDINGS: What is reassuringly normal (important to document)

━━━ STEP 2: ANATOMICAL-PATHOLOGICAL CORRELATION ━━━
For each abnormal finding, reason through:
- What pathological process could produce this signal/structural change?
- Is the distribution focal, multifocal, or diffuse?
- Is it unilateral or bilateral? Symmetric or asymmetric?
- Does it respect anatomical boundaries (vascular territory, lobe) or cross them?
  (Crossing the midline → think glioblastoma, lymphoma; respecting vascular territory → ischemia)
- Is there mass effect? (suggests acute process, neoplasm)
- Is there restricted diffusion pattern? (suggests acute ischemia, abscess, hypercellular tumor)

━━━ STEP 3: DIFFERENTIAL DIAGNOSIS (ranked) ━━━
Using the VINDICATE mnemonic framework, consider and RANK the top 3 differentials:
  V - Vascular (stroke, hemorrhage, AVM, cavernoma)
  I - Infectious/Inflammatory (abscess, encephalitis, MS, ADEM)
  N - Neoplastic (glioma grades I-IV, meningioma, metastasis, lymphoma)
  D - Degenerative (Alzheimer's, Parkinson's, MSA)
  I - Idiopathic / Iatrogenic
  C - Congenital / Developmental
  A - Autoimmune (lupus, vasculitis, NMOSD)
  T - Traumatic (contusion, DAI, SDH)
  E - Endocrine / Metabolic (hepatic encephalopathy, osmotic demyelination)

For your TOP 3 differentials, provide:
  1. [Most likely diagnosis]: Reasoning — what features support this? What features argue against?
  2. [Second differential]: Reasoning
  3. [Third differential / cannot exclude]: Reasoning

━━━ STEP 4: CLINICAL URGENCY TRIAGE ━━━
Classify using standard radiology urgency tiers:
  🔴 STAT / CRITICAL — Call radiologist immediately (herniation, hemorrhage, acute stroke, abscess with mass effect)
  🟠 URGENT — Report within hours (new significant finding, suspected malignancy)
  🟡 ROUTINE — Standard reporting timeline (chronic changes, stable findings)
  🟢 INCIDENTAL / NORMAL VARIANT — No immediate action needed

Justify your urgency classification.

━━━ STEP 5: WHAT THIS SCAN CANNOT TELL US ━━━
Be explicit about limitations:
- Which additional MRI sequences would add diagnostic value? (e.g., DWI for ischemia, FLAIR for MS, T1+Gd for BBB breakdown, MRS for tumor metabolism, SWI for microhemorrhages)
- Would clinical correlation change interpretation? How?
- Would comparison with prior imaging change interpretation?
- Any technical limitations that reduce diagnostic confidence?

━━━ STEP 6: CONFIDENCE ASSESSMENT ━━━
Overall confidence: HIGH / MEDIUM / LOW
Reasoning for this rating."""
    
    try:
        response = text_llm.invoke(prompt)
        reasoning = response.content
        
        # Extract confidence level for state
        confidence = _extract_confidence(reasoning)
        
        return {
            "reasoning_chain": reasoning,
            "differential_notes": "See reasoning chain above.",
            "confidence_level": confidence,
        }
    except Exception as e:
        return {"error": f"Reasoning Agent failed: {str(e)}"}


def _extract_confidence(reasoning_text: str) -> str:
    """Helper: extracts the overall confidence rating from reasoning."""
    text_upper = reasoning_text.upper()
    if "CONFIDENCE: HIGH" in text_upper or "CONFIDENCE ASSESSMENT: HIGH" in text_upper:
        return "HIGH"
    elif "CONFIDENCE: LOW" in text_upper or "CONFIDENCE ASSESSMENT: LOW" in text_upper:
        return "LOW"
    else:
        return "MEDIUM"


# ═══════════════════════════════════════════════════════
# AGENT 4: REPORT WRITER AGENT
# Job: Take all the messy agent outputs and write a clean,
# structured radiology-style report.
# This is what a real radiologist would sign off on.
# ═══════════════════════════════════════════════════════

def report_writer_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 4: Report Generation
    
    Synthesizes all previous agent outputs into a professional,
    structured radiology report format.
    """
    print("📝 [Agent 5/7] Report Writer Agent running...")
    
    if state.get("error"):
        # Even with errors, write a partial report
        return {
            "draft_report": f"REPORT GENERATION FAILED\n\nError: {state.get('error')}",
            "final_report": f"REPORT GENERATION FAILED\n\nError: {state.get('error')}",
        }
    
    text_llm = get_text_llm()
    
    context = f"""
PIPELINE OUTPUTS FOR REPORT GENERATION
=======================================

IMAGE DESCRIPTION:
{state.get('image_description', 'Not available')}

CLINICAL FINDINGS:
{state.get('findings', 'Not available')}

REGIONS OF CONCERN:
{state.get('regions_of_concern', 'None')}

REASONING CHAIN:
{state.get('reasoning_chain', 'Not available')}

CONFIDENCE LEVEL: {state.get('confidence_level', 'MEDIUM')}

PATIENT CONTEXT: {state.get('patient_context', 'Not provided')}
"""
    
    prompt = f"""{context}

You are a board-certified neuroradiologist dictating a formal structured report following ACR (American College of Radiology) reporting guidelines.

Write the complete report using EXACTLY this structure. Every section is mandatory — if information is unavailable, state "Unable to assess — [reason]."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRAIN MRI STRUCTURED ANALYSIS REPORT
AI-Assisted Academic Pipeline | NOT FOR CLINICAL USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATIENT & STUDY INFORMATION
────────────────────────────
Clinical Indication  : [Extract from patient context, or "Not provided"]
Relevant History     : [Extract from patient context, or "Not provided"]
MRI Sequence         : [From preprocessor — T1/T2/FLAIR/DWI/etc.]
Imaging Plane        : [From preprocessor — axial/coronal/sagittal]
Overall Confidence   : [HIGH / MEDIUM / LOW]
Urgency Tier         : [🔴 STAT / 🟠 URGENT / 🟡 ROUTINE / 🟢 INCIDENTAL]

TECHNIQUE
──────────
[1-2 sentences describing the apparent acquisition. E.g.: "Single-sequence brain MRI
in the [plane] plane, appearing consistent with [sequence] weighting based on signal
characteristics. [Note any quality limitations]."]

FINDINGS
─────────
Report findings by anatomical region. Use standard radiological descriptors.
Mark uncertain findings as *(low confidence)*.

Extra-Axial Compartment:
[findings]

Cerebral Cortex & Sulci:
[findings]

White Matter:
[findings]

Deep Gray Matter (Basal Ganglia / Thalami):
[findings]

Ventricular System:
[findings]

Corpus Callosum & Midline:
[findings]

Posterior Fossa (Cerebellum / Brainstem):
[findings]

Vascular Structures:
[findings]

Focal Lesions (if present):
[For each: location, size estimate, signal characteristics, margins, mass effect, edema]

IMPRESSION
───────────
[This is the most important section — what clinicians act on.]
[3-6 numbered lines, most significant finding first.]
[Be specific, avoid vague language like "cannot exclude." Instead: "findings are
 most consistent with X; Y remains in the differential and would require Z to exclude."]

1.
2.
3.

DIFFERENTIAL DIAGNOSIS
───────────────────────
Most Likely  : [diagnosis] — [one-line reasoning]
Consider Also: [diagnosis] — [one-line reasoning]
Less Likely  : [diagnosis] — [why still in differential]

RECOMMENDATIONS
────────────────
[Specific, actionable, ranked by priority]
1. [Most important recommendation]
2. [Additional imaging if needed — specify sequence and reason]
3. [Clinical correlation recommendation]
4. [Follow-up timeline if applicable]

ADDITIONAL SEQUENCES THAT WOULD ADD DIAGNOSTIC VALUE
──────────────────────────────────────────────────────
[List sequences not performed that would help, with specific clinical rationale]
e.g.: "DWI/ADC — to evaluate for acute restricted diffusion / ischemia"
      "T1 post-gadolinium — to assess blood-brain barrier integrity / enhancement"
      "MR Spectroscopy — to characterize metabolite profile of focal lesion"
      "SWI — to detect microhemorrhages or calcifications"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACADEMIC DISCLAIMER
This report was generated by a multi-agent AI pipeline developed for a
Master's-level Agentic AI research project. It is strictly for academic
and educational demonstration purposes. This output has NOT been validated
for clinical accuracy and MUST NOT be used for medical diagnosis, treatment
planning, or any clinical decision-making. All findings require independent
verification by a board-certified radiologist with access to the original
DICOM data and full clinical context.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now generate the complete report filling in all sections based on the pipeline outputs above."""
    
    try:
        response = text_llm.invoke(prompt)
        report = response.content
        
        return {
            "draft_report": report,
            "final_report": report,  # Critic agent will refine this
        }
    except Exception as e:
        return {"error": f"Report Writer Agent failed: {str(e)}"}


# ═══════════════════════════════════════════════════════
# AGENT 5 (BONUS): CRITIC AGENT
# Job: Review the draft report and flag any issues.
# This is the "self-reflection" loop — agentic AI reviewing its own work.
# Makes your project stand out because it shows metacognition.
# ═══════════════════════════════════════════════════════

def critic_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 5 (Bonus): Self-Critique
    
    Reviews the draft report for:
    - Overconfident claims
    - Internal inconsistencies  
    - Missing information
    - Safety issues (claims that could mislead)
    
    Then outputs an improved final report.
    
    This agent demonstrates a key agentic pattern: REFLECTION.
    """
    print("🔬 [Agent 6/7] Critic Agent running...")
    
    if state.get("error") or not state.get("draft_report"):
        return {}
    
    text_llm = get_text_llm()
    
    prompt = f"""You are a neuroradiology quality assurance specialist and senior attending reviewing an AI-generated brain MRI report before it leaves the department.

DRAFT REPORT:
{state.get('draft_report')}

RAW FINDINGS FROM ANALYSIS AGENT (ground truth reference):
{state.get('findings', 'Not available')}

REASONING CHAIN:
{state.get('reasoning_chain', 'Not available')}

Perform a rigorous QA review checking for these specific failure modes in radiology AI reports:

━━━ QA CHECKLIST ━━━

1. FACTUAL ACCURACY
   □ Does the Impression match the Findings section? (contradictions = critical error)
   □ Are laterality statements correct and consistent? (left vs right errors are dangerous)
   □ Are anatomical terms used correctly?
   □ Are signal descriptors (hyperintense/hypointense) used correctly for the stated sequence?

2. OVERCONFIDENCE AUDIT
   □ Any diagnostic conclusions stated as certainties that should be hedged?
   □ Any findings described as "normal" when they were actually "unable to assess"?
   □ Are confidence levels appropriate for a single-sequence, single-slice analysis?
   □ Does the differential appropriately reflect uncertainty?

3. COMPLETENESS CHECK
   □ Does the Impression address the clinical indication stated in the patient history?
   □ Are follow-up recommendations specific and actionable (not just "clinical correlation")?
   □ Are the most clinically urgent findings listed FIRST in the Impression?
   □ Is the additional sequences section specific and clinically reasoned?

4. SAFETY FLAGS
   □ Any statement that could cause a reader to dismiss a potentially serious finding?
   □ Any missing urgent/STAT tier recommendation that should be present?
   □ Is the academic disclaimer clearly present and unambiguous?

5. RADIOLOGICAL LANGUAGE QUALITY
   □ Is language precise (avoid "possible", "maybe" — use "cannot exclude", "warrants consideration")?
   □ Are size estimates provided where possible?
   □ Are lesion descriptors complete (location, signal, margins, mass effect)?

━━━ YOUR RESPONSE FORMAT ━━━

CRITIQUE NOTES:
[List every issue found, numbered. For each: what is wrong, why it matters, how to fix it.
If no issues found in a category, write "✓ Pass"]

---REVISED FINAL REPORT---
[Full corrected report — same structure, all issues fixed. If the draft was good, reproduce it with only targeted corrections.]"""
    
    try:
        response = text_llm.invoke(prompt)
        full_response = response.content
        
        # Split critique notes from revised report
        if "---REVISED FINAL REPORT---" in full_response:
            parts = full_response.split("---REVISED FINAL REPORT---")
            critique = parts[0].replace("CRITIQUE NOTES:", "").strip()
            final = parts[1].strip()
        else:
            critique = "Critic agent response could not be parsed."
            final = state.get("draft_report", "")
        
        return {
            "critique_notes": critique,
            "final_report": final,
        }
    except Exception as e:
        # If critic fails, just use the draft report
        return {
            "critique_notes": f"Critic agent failed: {str(e)}. Using draft report.",
            "final_report": state.get("draft_report", ""),
        }


# ═══════════════════════════════════════════════════════
# AGENT 6: TUMOR CONCLUSION AGENT
# Job: Runs LAST. Reads everything the previous agents produced
# and renders a single, clear tumor verdict.
#
# Why a separate agent for this instead of burying it in the report?
#   - Forced focus: the agent's ONLY job is this one question
#   - Explicit reasoning: it must justify its verdict from evidence
#   - Clean UI: we can display the verdict as a prominent badge
#   - Separation of concerns: report = full picture, verdict = bottom line
# ═══════════════════════════════════════════════════════

def tumor_conclusion_agent(state: MRIAnalysisState) -> dict:
    """
    Agent 6: Tumor Verdict

    Synthesizes all prior agent outputs and renders a final
    tumor assessment: DETECTED, NOT DETECTED, or UNCERTAIN.

    This runs after the critic so it has access to the most
    refined, reviewed version of all findings.
    """
    print("🔎 [Agent 7/7] Tumor Conclusion Agent running...")

    if state.get("error") or not state.get("final_report"):
        return {
            "tumor_conclusion": "UNCERTAIN — Pipeline did not complete successfully. Manual review required."
        }

    text_llm = get_text_llm()

    prompt = f"""You are a neuroradiology specialist rendering a final tumor assessment verdict.

You have access to the complete analysis from a multi-agent brain MRI pipeline:

SYSTEMATIC FINDINGS:
{state.get('findings', 'Not available')}

REASONING CHAIN & DIFFERENTIALS:
{state.get('reasoning_chain', 'Not available')}

FINAL REVIEWED REPORT:
{state.get('final_report', 'Not available')}

OVERALL CONFIDENCE LEVEL: {state.get('confidence_level', 'MEDIUM')}

━━━ YOUR TASK ━━━

Based ONLY on the evidence above, render a tumor verdict using EXACTLY this format:

VERDICT: [TUMOR DETECTED / TUMOR NOT DETECTED / UNCERTAIN]

SUPPORTING EVIDENCE:
- [bullet 1: specific finding that supports your verdict]
- [bullet 2: specific finding that supports your verdict]
- [bullet 3 if applicable]

AGAINST EVIDENCE (findings that argue against your verdict, if any):
- [bullet or "None"]

VERDICT CONFIDENCE: [HIGH / MEDIUM / LOW]

WHAT WOULD CHANGE THIS VERDICT:
[One sentence: what additional imaging, sequences, or clinical info could upgrade or overturn this verdict]

━━━ VERDICT DEFINITIONS ━━━
- TUMOR DETECTED: One or more findings are strongly consistent with a neoplastic process (mass lesion, ring enhancement, surrounding edema, mass effect, abnormal signal with neoplastic pattern)
- TUMOR NOT DETECTED: No findings suggest neoplasm; other diagnoses (vascular, inflammatory, degenerative) better explain the findings, or scan appears normal
- UNCERTAIN: Findings are ambiguous, image quality is insufficient, or additional sequences are required before a neoplastic process can be confidently included or excluded

Be honest. UNCERTAIN is a valid and responsible verdict — do not force a binary answer when the evidence doesn't support one."""

    try:
        response = text_llm.invoke(prompt)
        conclusion = response.content.strip()
        return {"tumor_conclusion": conclusion}

    except Exception as e:
        return {
            "tumor_conclusion": f"UNCERTAIN — Tumor conclusion agent failed: {str(e)}"
        }
