"""
agents/fir_processor.py
──────────────────────────────────────────────────────────────
FIR/Complaint Photo Processing Agent

Purpose:
- Takes FIR photo (JPG/PNG) or PDF
- Extracts facts using Gemini Vision
- Retrieves applicable laws (BNS, IPC, CrPC)
- Assesses legal risk (bail eligibility, custody likelihood)
- Auto-generates 4 legal documents (complaint, bail app, petition, cover letter)
- Produces next-steps checklist

Usage:
    from agents.fir_processor import FIRProcessor
    
    processor = FIRProcessor()
    file_bytes = open("fir_photo.jpg", "rb").read()
    result = processor.process(file_bytes, mime_type="image/jpeg")
    
    print(result["facts_extracted"])
    print(result["laws_identified"])
    print(result["risk_assessment"])
"""

import json
from datetime import datetime
from config import client, MODEL_NAME
from agents.retrieval import RetrievalAgent
from google.genai import types
import fitz  # PyMuPDF - better text rendering than FPDF
import io


class FIRProcessor:
    """
    Processes FIR photos/PDFs and generates legal documents.
    
    Pipeline:
    1. Extract facts (Gemini Vision)
    2. Normalize facts (standardize locations, crime types)
    3. Retrieve laws (FAISS search)
    4. Assess risk (bail eligibility, custody likelihood)
    5. Generate documents (complaint, bail, petition, cover letter)
    6. Generate next steps checklist
    """
    
    def __init__(self):
        print("[FIRProcessor] Initializing...")
        self.retrieval = RetrievalAgent()
        print("[FIRProcessor] Ready.\n")
    
    def process(self, file_bytes, mime_type):
        """
        Main entry point for FIR processing.
        
        Args:
            file_bytes (bytes): Image/PDF file content
            mime_type (str): "image/jpeg", "image/png", or "application/pdf"
        
        Returns:
            dict: Comprehensive results including:
                - facts_extracted: Raw Gemini Vision output
                - facts_normalized: Standardized facts
                - laws_identified: List of applicable legal sections
                - risk_assessment: Bail eligibility, custody risk, etc.
                - documents: Text drafts of 4 legal documents
                - pdfs: PDF bytes for all 4 documents
                - next_steps: Actionable checklist
        """
        
        print("[FIRProcessor] Starting processing...")
        
        # STAGE 1: Extract facts from image using Vision
        facts = self._extract_facts(file_bytes, mime_type)
        if "error" in facts:
            return {"error": facts["error"]}
        
        # STAGE 2: Normalize facts (standardize values)
        normalized = self._normalize_facts(facts)
        
        # STAGE 3: Retrieve applicable laws from FAISS
        laws = self._retrieve_laws(normalized)
        
        # STAGE 4: Assess legal risk
        risk = self._assess_risk(normalized, laws)
        
        # STAGE 5: Generate document texts
        doc_texts = self._generate_document_texts(normalized, laws, risk)
        
        # STAGE 6: Convert texts to PDFs
        doc_pdfs = self._generate_pdfs(doc_texts)
        
        # STAGE 7: Generate next steps
        next_steps = self._generate_next_steps(normalized, risk)
        
        print("[FIRProcessor] ✅ Processing complete.\n")
        
        return {
            "facts_extracted": facts,
            "facts_normalized": normalized,
            "laws_identified": laws,
            "risk_assessment": risk,
            "documents": doc_texts,
            "pdfs": doc_pdfs,
            "next_steps": next_steps,
        }
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 1: Extract facts from image using Gemini Vision
    # ═════════════════════════════════════════════════════════════
    
    def _extract_facts(self, file_bytes, mime_type):
        """Use Gemini Vision to read FIR photo and extract key facts."""
        print("[FIRProcessor] Stage 1: Extracting facts from image...")
        
        prompt = """You are a legal document analyzer. 
        Read this FIR (First Information Report) or police complaint photo carefully.
        Extract and return ONLY a valid JSON object with these fields.
        For any field you cannot see clearly, use "Not Visible" instead of guessing.
        
        REQUIRED JSON FORMAT - DO NOT DEVIATE:
        {
            "fir_number": "string",
            "date_of_fir": "string (YYYY-MM-DD format)",
            "police_station": "string",
            "location": "string (city/state)",
            "accused_name": "string or 'Unknown'",
            "accused_details": "string (age, occupation, etc)",
            "victim_name": "string or 'Not disclosed'",
            "victim_details": "string (age, gender, occupation)",
            "crime_type": "string (e.g., molestation, theft, fraud)",
            "incident_date": "string (YYYY-MM-DD format if visible)",
            "incident_location": "string",
            "incident_narration": "string (2-3 sentences summary)",
            "investigating_officer": "string or 'Not mentioned'",
            "status": "string (registered, under investigation, etc)"
        }
        
        IMPORTANT RULES:
        - Return ONLY the JSON object, NO other text
        - Do NOT use markdown code blocks
        - Do NOT include explanations
        - If field is unclear, set to "Not Visible"
        - Use exactly these field names
        """
        
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    prompt
                ]
            )
            
            # Parse JSON response
            json_str = resp.text.strip()
            
            # Clean up markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            facts = json.loads(json_str)
            print(f"[FIRProcessor] ✓ Extracted {len(facts)} fields from image")
            return facts
        
        except json.JSONDecodeError as e:
            print(f"[FIRProcessor] ❌ JSON parsing error: {e}")
            return {"error": f"Could not parse document. Please ensure it's a clear FIR photo: {str(e)}"}
        except Exception as e:
            print(f"[FIRProcessor] ❌ Vision extraction error: {e}")
            return {"error": f"Vision API error: {str(e)}"}
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 2: Normalize facts (standardize locations, crime types)
    # ═════════════════════════════════════════════════════════════
    
    def _normalize_facts(self, facts):
        """Standardize extracted facts for law retrieval and reasoning."""
        print("[FIRProcessor] Stage 2: Normalizing facts...")
        
        normalized = {
            "fir_number": facts.get("fir_number", "Unknown"),
            "location": self._normalize_location(facts.get("location", "")),
            "crime_type": self._normalize_crime_type(facts.get("crime_type", "")),
            "incident_date": facts.get("incident_date", "Unknown"),
            "has_accused": "unknown" not in facts.get("accused_name", "Unknown").lower(),
            "victim_gender": self._extract_gender(facts.get("victim_details", "")),
            "incident_narration": facts.get("incident_narration", ""),
            "status": facts.get("status", "registered"),
        }
        
        print(f"[FIRProcessor] ✓ Normalized: {normalized['location']} | {normalized['crime_type']}")
        return normalized
    
    def _normalize_location(self, location_str):
        """Convert location to standard state format."""
        state_map = {
            "delhi": "Delhi",
            "mumbai": "Maharashtra",
            "bangalore": "Karnataka",
            "noida": "Delhi-NCR",
            "gurgaon": "Delhi-NCR",
            "kolkata": "West Bengal",
            "hyderabad": "Telangana",
            "pune": "Maharashtra",
            "jaipur": "Rajasthan",
            "ahmedabad": "Gujarat",
            "chennai": "Tamil Nadu",
            "surat": "Gujarat",
            "lucknow": "Uttar Pradesh",
        }
        
        location_lower = location_str.lower()
        for key, value in state_map.items():
            if key in location_lower:
                return value
        
        return location_str if location_str and location_str != "Not Visible" else "Unknown"
    
    def _normalize_crime_type(self, crime_str):
        """Standardize crime type naming."""
        crime_map = {
            "molestation": "sexual_harassment",
            "rape": "sexual_assault",
            "theft": "theft",
            "fraud": "cheating",
            "assault": "physical_assault",
            "harassment": "harassment",
            "dowry": "dowry_related",
            "extortion": "extortion",
            "kidnapping": "kidnapping",
            "cheating": "cheating",
        }
        
        crime_lower = crime_str.lower()
        for key, value in crime_map.items():
            if key in crime_lower:
                return value
        
        return crime_str if crime_str and crime_str != "Not Visible" else "other_crime"
    
    def _extract_gender(self, details_str):
        """Extract victim gender from details."""
        details_lower = details_str.lower()
        if any(w in details_lower for w in ["woman", "female", "girl", "she"]):
            return "female"
        elif any(w in details_lower for w in ["man", "male", "boy", "he"]):
            return "male"
        return "unspecified"
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 3: Retrieve applicable laws using FAISS
    # ═════════════════════════════════════════════════════════════
    
    def _retrieve_laws(self, normalized):
        """Use FAISS to find relevant legal sections."""
        print("[FIRProcessor] Stage 3: Retrieving applicable laws...")
        
        # Build search query from normalized facts
        search_query = f"{normalized['crime_type']} {normalized['location']} {normalized['incident_narration'][:300]}"
        
        # Use existing RetrievalAgent
        chunks = self.retrieval.retrieve(search_query)
        
        laws = []
        for chunk in chunks[:5]:  # Top 5 most relevant
            laws.append({
                "section": chunk.get("act", "Unknown"),
                "content_preview": chunk.get("content", "")[:300],
                "relevance_score": chunk.get("relevance", 0.0),
                "act_type": self._classify_act(chunk.get("act", "")),
            })
        
        print(f"[FIRProcessor] ✓ Retrieved {len(laws)} applicable legal sections")
        return laws
    
    def _classify_act(self, act_name):
        """Classify which type of act (BNS, IPC, CrPC)."""
        act_lower = act_name.lower()
        if "bns" in act_lower or "bharatiya" in act_lower:
            return "BNS"
        elif "ipc" in act_lower:
            return "IPC"
        elif "crpc" in act_lower or "code of criminal" in act_lower:
            return "CrPC"
        else:
            return "Other"
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 4: Assess legal risk
    # ═════════════════════════════════════════════════════════════
    
    def _assess_risk(self, normalized, laws):
        """Assess bail eligibility and custody likelihood."""
        print("[FIRProcessor] Stage 4: Assessing legal risk...")
        
        # Rule-based risk assessment
        bail_eligible = True
        custody_risk = 30  # Default risk %
        
        crime_type = normalized["crime_type"]
        
        # Adjust based on crime severity
        if crime_type in ["sexual_assault", "murder", "kidnapping"]:
            bail_eligible = False
            custody_risk = 85
        elif crime_type in ["sexual_harassment", "harassment", "cheating"]:
            custody_risk = 35
        elif crime_type in ["theft", "assault"]:
            custody_risk = 45
        elif crime_type == "dowry_related":
            custody_risk = 70
        
        # Increase risk if victim is female in sensitive crimes
        if normalized["victim_gender"] == "female" and \
           crime_type in ["sexual_assault", "sexual_harassment", "dowry_related"]:
            custody_risk = min(custody_risk + 15, 95)
        
        # Categorize risk level
        if custody_risk > 70:
            risk_level = "HIGH"
            recommendation = "⚠️ Custody likely. File anticipatory bail immediately with strong grounds."
        elif custody_risk > 40:
            risk_level = "MODERATE"
            recommendation = "Moderate custody risk. Prepare bail documents and financial sureties."
        else:
            risk_level = "LOW"
            recommendation = "Low custody risk. Standard bail procedures should work."
        
        result = {
            "bail_eligible": bail_eligible,
            "custody_risk_percentage": custody_risk,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "note": "Assessment based on crime type and available information. Consult lawyer for personalized advice."
        }
        
        print(f"[FIRProcessor] ✓ Risk assessment: {risk_level}")
        return result
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 5: Generate document texts (before PDF conversion)
    # ═════════════════════════════════════════════════════════════
    
    def _generate_document_texts(self, normalized, laws, risk):
        """Generate text content for 4 legal documents."""
        print("[FIRProcessor] Stage 5: Generating document texts...")
        
        # Format laws for citations
        law_citations = "\n".join([f"• {law['section']}" for law in laws[:3]])
        
        documents = {
            "complaint": self._template_police_complaint(normalized, law_citations),
            "bail": self._template_bail_application(normalized, law_citations, risk),
            "petition": self._template_court_petition(normalized, law_citations),
            "cover_letter": self._template_advocate_letter(normalized),
        }
        
        print("[FIRProcessor] ✓ Generated text for 4 documents")
        return documents
    
    def _template_police_complaint(self, normalized, citations):
        """Template for police complaint."""
        return f"""
FORMAL COMPLAINT / FIR REFERENCE LETTER
{'='*60}

Ref: FIR No. {normalized['fir_number']}
Date: {datetime.now().strftime('%d-%m-%Y')}
Location: {normalized['location']}

TO THE ESTEEMED POLICE COMMISSIONER / STATION OFFICER,
[Police Station Name], {normalized['location']}

SUBJECT: Formal Complaint in connection with FIR No. {normalized['fir_number']}

Dear Sir/Madam,

I hereby lodge this formal complaint/reference regarding the incident 
dated {normalized['incident_date']} at {normalized['location']} concerning 
criminal activities as detailed below:

═══════════════════════════════════════════════════════════════

FACTS OF THE CASE:

{normalized['incident_narration']}

Crime Type: {normalized['crime_type'].replace('_', ' ').title()}
Status: {normalized['status'].title()}

═══════════════════════════════════════════════════════════════

APPLICABLE LEGAL PROVISIONS:

This matter falls under the following legal framework:

{citations}

═══════════════════════════════════════════════════════════════

PRAYERS/REQUESTS:

1. Swift investigation into the above matter
2. Recovery and preservation of evidence
3. Immediate preventive action against accused
4. Regular updates on investigation progress
5. Proper documentation and filing of charges as per law

═══════════════════════════════════════════════════════════════

I request your immediate attention to this serious matter and 
assure full cooperation in the investigation process.

Respectfully submitted,

___________________________
Signature of Complainant
Date: {datetime.now().strftime('%d-%m-%Y')}
Name: [Your Name]
Address: [Your Address]
Contact: [Your Phone]

Note: This is an auto-generated legal template. Please consult a lawyer 
for personalized advice and to verify accuracy of specific details.
"""
    
    def _template_bail_application(self, normalized, citations, risk):
        """Template for bail application."""
        bail_status = "HIGH RISK - ANTICIPATORY BAIL RECOMMENDED" if not risk['bail_eligible'] else "Regular Bail Application"
        
        return f"""
BAIL APPLICATION UNDER CrPC SECTION 437/439
{'='*60}

Applicant: [YOUR NAME]
FIR Reference: {normalized['fir_number']}
Date: {datetime.now().strftime('%d-%m-%Y')}
Court: [District Court / High Court, {normalized['location']}]

STATUS: {bail_status}

═══════════════════════════════════════════════════════════════

I. NATURE OF OFFENCE

Crime Type: {normalized['crime_type'].replace('_', ' ').title()}
Custody Risk Assessment: {risk['custody_risk_percentage']}%
Risk Level: {risk['risk_level']}

═══════════════════════════════════════════════════════════════

II. LEGAL FRAMEWORK

This application is filed under Section 437/439 of the Code of 
Criminal Procedure, 1973. The following legal provisions apply:

{citations}

═══════════════════════════════════════════════════════════════

III. GROUNDS FOR BAIL/ANTICIPATORY BAIL

A) Personal Background:
   • Permanent resident of {normalized['location']}
   • Stable residence: [Your Address]
   • Employment: [Your Occupation]
   • Monthly Income: [Amount]
   
B) Criminal History:
   • No previous criminal convictions
   • No history of absconding
   • No involvement in violence
   • Clean police record

C) Bail Conditions Offered:
   • Unconditional surrender before court
   • Regular reporting to police station (as ordered)
   • Restriction on movements (as per bail conditions)
   • Non-interference with investigation/witnesses
   • Attendance at all court hearings

═══════════════════════════════════════════════════════════════

IV. SURETIES

Primary Surety:
   Name: [Surety Name 1]
   Relation: [Family/Friend]
   Address: [Full Address]
   Annual Income: [Amount]
   Occupation: [Profession]

Secondary Surety:
   Name: [Surety Name 2]
   Relation: [Family/Friend]
   Address: [Full Address]
   Annual Income: [Amount]
   Occupation: [Profession]

═════════════════════════════════════════════════════════════════

V. PRAYER/REQUEST

This Hon'ble Court is humbly requested to:

1. Accept this bail application
2. Grant bail upon furnishing bond and sureties
3. Impose reasonable bail conditions
4. Allow applicant to remain free pending trial
5. Pass such other orders as deemed fit

═════════════════════════════════════════════════════════════════

Respectfully submitted,

___________________________
Applicant Signature
Date: {datetime.now().strftime('%d-%m-%Y')}

Advocate/Representative: [Advocate Name & License]
Bar Council Reg. No.: [Number]

CERTIFICATION: This is an auto-generated legal document. 
Please have it reviewed and filed by your advocate.
"""
    
    def _template_court_petition(self, normalized, citations):
        """Template for court petition."""
        return f"""
PETITION TO THE HONORABLE DISTRICT COURT / HIGH COURT
{'='*60}

PETITION NO: [TO BE FILLED BY COURT]
Petitioner: [YOUR NAME]
Respondent: State of India & Others
Date: {datetime.now().strftime('%d-%m-%Y')}

RE: FIR No. {normalized['fir_number']} | {normalized['location']} Police Station

═════════════════════════════════════════════════════════════════

PRAYER/RELIEF SOUGHT:

This Petition is filed seeking the following relief:

1. Direction to the respondent (State) to produce all case records
   relating to FIR No. {normalized['fir_number']};

2. Order for expedited investigation and early filing of charge sheet;

3. Bail/anticipatory bail as per applicable law;

4. Any other relief that this Hon'ble Court deems just and proper.

═════════════════════════════════════════════════════════════════

STATEMENT OF FACTS:

The present petition arises out of FIR No. {normalized['fir_number']} 
registered at {normalized['location']} Police Station.

Briefly, the facts are as under:

Case Background:
{normalized['incident_narration']}

Current Status:
The matter is currently under investigation. Investigation has been 
ongoing since {normalized['incident_date']}.

═════════════════════════════════════════════════════════════════

GROUNDS FOR PETITION:

1. LEGAL BASIS:
   
   The following legal provisions are applicable:
   
{citations}

2. GROUNDS:
   
   a) In the interest of justice and to prevent unnecessary detention
   b) For early disposal and expedited hearing of the matter
   c) To safeguard the rights of the petitioner
   d) To ensure compliance with due process of law

═════════════════════════════════════════════════════════════════

RELIEFS SOUGHT:

This Hon'ble Court is respectfully requested to:

1. Admit this petition
2. Issue notice to respondents
3. Grant appropriate relief as prayed
4. Pass such other orders as deemed fit and proper

═════════════════════════════════════════════════════════════════

Respectfully submitted,

___________________________
Petitioner Signature
Date: {datetime.now().strftime('%d-%m-%Y')}

Advocate Name & License: [To be filled]
Contact: [Advocate Contact]

VERIFICATION: I, [Name], hereby verify that the contents of 
this petition are true to my knowledge and belief.
"""
    
    def _template_advocate_letter(self, normalized):
        """Template for advocate's cover letter."""
        return f"""
ADVOCATE/LAWYER COVER LETTER
{'='*60}

TO THE HONORABLE COURT,
[District Court / High Court]
{normalized['location']}

Date: {datetime.now().strftime('%d-%m-%Y')}

RE: FIR NO. {normalized['fir_number']} | Petitioner: [Name]

═════════════════════════════════════════════════════════════════

Dear Hon'ble Judge,

I respectfully submit the annexed documents on behalf of my client 
in connection with the above-captioned matter for your kind perusal.

═════════════════════════════════════════════════════════════════

BRIEF BACKGROUND:

A case was registered under FIR No. {normalized['fir_number']} at 
{normalized['location']} Police Station relating to the following facts:

INCIDENT NATURE: {normalized['crime_type'].replace('_', ' ').title()}
DATE OF INCIDENT: {normalized['incident_date']}
LOCATION: {normalized['location']}

Summary: {normalized['incident_narration'][:300]}

═════════════════════════════════════════════════════════════════

DOCUMENTS ANNEXED:

This petition is accompanied by the following supporting documents:

1. ☐ Certified copy of FIR (attached)
2. ☐ Complaint letter (auto-generated, attached)
3. ☐ Bail application with sureties (attached)
4. ☐ Identity and address proof of petitioner
5. ☐ Affidavit and personal undertaking
6. ☐ Certificate of investigation status
7. ☐ Witness statements (if available)

═════════════════════════════════════════════════════════════════

RELIEF SOUGHT:

My client hereby prays for:

1. Expedited hearing and early justice in this matter
2. Bail/anticipatory bail as per applicable law
3. Direction for completion of investigation
4. Any such relief as the Hon'ble Court deems fit

═════════════════════════════════════════════════════════════════

ARGUMENTS:

[Your advocate will brief the court with relevant case law 
and constitutional provisions during oral hearing]

═════════════════════════════════════════════════════════════════

I am available for any clarifications or further submissions that 
the Hon'ble Court may require.

Thanking you for your kind attention.

Respectfully submitted,

___________________________
[Advocate Name]
License No.: [Bar Council Registration]
Contact: [Phone/Email]

COURT STAMP SPACE:
[Advocate to affix stamp here]

═════════════════════════════════════════════════════════════════

NOTE: This is an auto-generated legal document prepared by AI Legal 
Assistant. It must be reviewed, verified, and filed by a qualified 
advocate as per rules of the court.
"""
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 6: Convert texts to PDF
    # ═════════════════════════════════════════════════════════════
    
    def _generate_pdfs(self, doc_texts):
        """Convert document texts to PDF bytes."""
        print("[FIRProcessor] Stage 6: Converting to PDF format...")
        
        pdfs = {
            "complaint": self._text_to_pdf("Police Complaint", doc_texts["complaint"]),
            "bail": self._text_to_pdf("Bail Application", doc_texts["bail"]),
            "petition": self._text_to_pdf("Court Petition", doc_texts["petition"]),
            "cover_letter": self._text_to_pdf("Advocate Cover Letter", doc_texts["cover_letter"]),
        }
        
        print("[FIRProcessor] ✓ Generated 4 PDF documents")
        return pdfs
    
    def _text_to_pdf(self, title, text):
        """Convert text to PDF bytes using PyMuPDF (much more robust than FPDF)."""
        try:
            # Create a new PDF document
            doc = fitz.open()
            page = doc.new_page()  # Create new page without perm arg
            
            # Set up text properties
            title_font_size = 14
            subtitle_font_size = 9
            body_font_size = 10
            
            # Margins
            left_margin = 36  # ~0.5 inch
            right_margin = 36
            top_margin = 36
            line_height = 12
            
            # Add title
            title_rect = fitz.Rect(left_margin, top_margin, page.rect.width - right_margin, top_margin + 30)
            page.insert_textbox(
                title_rect,
                title,
                fontname="helv",
                fontsize=title_font_size,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            # Add subtitle with generation date
            subtitle_y = top_margin + 35
            subtitle_text = f"Generated on {datetime.now().strftime('%d-%m-%Y %H:%M')}"
            subtitle_rect = fitz.Rect(left_margin, subtitle_y, page.rect.width - right_margin, subtitle_y + 15)
            page.insert_textbox(
                subtitle_rect,
                subtitle_text,
                fontname="helv",
                fontsize=subtitle_font_size,
                color=(0.4, 0.4, 0.4),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            # Process body text
            current_y = subtitle_y + 25
            max_y = page.rect.height - 40  # Leave bottom margin
            content_width = page.rect.width - left_margin - right_margin
            
            for line in text.split("\n"):
                cleaned_line = line.strip()
                
                # Skip decorator lines
                if cleaned_line.startswith("═") or cleaned_line.startswith("─"):
                    current_y += 3
                    continue
                
                if not cleaned_line:
                    current_y += 6
                    continue
                
                # Clean problematic characters while keeping legal content
                safe_text = cleaned_line.replace("═", "").replace("─", "").replace("☐", "□")
                
                # Safety: truncate extremely long lines
                if len(safe_text) > 200:
                    safe_text = safe_text[:197] + "..."
                
                try:
                    # Create text box for this line
                    line_rect = fitz.Rect(
                        left_margin, 
                        current_y, 
                        page.rect.width - right_margin, 
                        current_y + 100  # Let it expand as needed
                    )
                    
                    # Insert text and get the actual height used
                    unused_space = page.insert_textbox(
                        line_rect,
                        safe_text,
                        fontname="helv",
                        fontsize=body_font_size,
                        color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_LEFT
                    )
                    
                    used_height = 100 - unused_space if unused_space > 0 else 100
                    current_y += used_height + 2  # Add small spacing between lines
                    
                except Exception as e:
                    print(f"[FIRProcessor] ⚠️ Line skipped: {str(e)[:40]}")
                    current_y += 12
                
                # Check if we need a new page
                if current_y > max_y:
                    page = doc.new_page()
                    current_y = top_margin
            
            # Add footer on last page
            footer_y = page.rect.height - 25
            footer_text = "Generated by AI Legal Assistant | Consult a qualified advocate before filing"
            footer_rect = fitz.Rect(left_margin, footer_y, page.rect.width - right_margin, page.rect.height - 5)
            page.insert_textbox(
                footer_rect,
                footer_text,
                fontname="helv",
                fontsize=8,
                color=(0.5, 0.5, 0.5),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            # Get PDF as bytes
            pdf_bytes = doc.write()
            doc.close()
            
            return pdf_bytes
        
        except Exception as e:
            print(f"[FIRProcessor] ⚠️ PDF generation error: {e}")
            # Fallback: create a simple PDF with text content notification
            try:
                doc = fitz.open()
                page = doc.new_page()
                
                text_content = f"{title}\n\nDocument generated with text content.\nPlease use the text version if PDF rendering has issues.\n\nGenerated: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
                page.insert_text(
                    (36, 36),
                    text_content,
                    fontname="helv",
                    fontsize=11,
                    color=(0, 0, 0)
                )
                
                pdf_bytes = doc.write()
                doc.close()
                return pdf_bytes
            except:
                # Last resort: return empty bytes
                return b""
    
    # ═════════════════════════════════════════════════════════════
    # STAGE 7: Generate next steps checklist
    # ═════════════════════════════════════════════════════════════
    
    def _generate_next_steps(self, normalized, risk):
        """Generate actionable next steps checklist."""
        print("[FIRProcessor] Stage 7: Generating next steps checklist...")
        
        urgency = "🔴 URGENT" if risk['risk_level'] == 'HIGH' else "🟡 MODERATE" if risk['risk_level'] == 'MODERATE' else "🟢 STANDARD"
        
        next_steps = f"""
✅ IMMEDIATE ACTION CHECKLIST - FIR NO. {normalized['fir_number']}
{'='*70}

URGENCY LEVEL: {urgency}
Risk Assessment: {risk['risk_level']} ({risk['custody_risk_percentage']}% custody risk)

═══════════════════════════════════════════════════════════════════════

🔴 TODAY (WITHIN 24 HOURS) - CRITICAL:

☐ Verify FIR has been officially registered at police station
☐ Obtain FIR receipt/acknowledgment copy (very important)
☐ Collect FIR reference number and station contact details
☐ Contact a qualified criminal lawyer in {normalized['location']}
☐ Give complete problem statement to lawyer
☐ If high-risk: File for anticipatory bail TODAY (don't delay)
☐ Inform trusted family members and arrange bail funds if needed
☐ Preserve all evidence (messages, photos, documents, witnesses)
☐ Call NALSA (National Legal Services) - 15100 for free legal advice

═══════════════════════════════════════════════════════════════════════

🟡 WITHIN 3-7 DAYS - IMPORTANT:

☐ Meet your lawyer with all case documents
☐ Prepare bail application with financial details of sureties
☐ Gather identity proof, address proof, income certificates
☐ Collect contact details of 2-3 potential sureties (trusted people)
☐ File bail application to nearest District Court
☐ Prepare financial guarantee (usually ₹50,000 - ₹5,00,000 range)
☐ Request investigation status update from police
☐ Document any harassment or witness tampering attempts
☐ File complaint if harassment continues

═══════════════════════════════════════════════════════════════════════

🟢 FIRST COURT HEARING - PREPARATION:

☐ Attend court with your lawyer (DO NOT MISS)
☐ Bring all documents: ID, address proof, bail application, sureties
☐ Bring sureties with their documents if required by court
☐ Be calm and truthful during court questioning
☐ Follow all instructions given by judge
☐ Request adjournment if you need more time for preparation
☐ Ask for copy of charge sheet when filed
☐ Get bail bond details in writing from court

═══════════════════════════════════════════════════════════════════════

📞 KEY CONTACTS FOR {normalized['location']} / YOUR STATE:

LEGAL SERVICES:
• NALSA (National Legal Services Authority): 15100 (Free Legal Aid)
• State Legal Services: [Check your state bar association website]
• District Court: [Local court website]
• {normalized['location']} Police Commissioner: [Search online]

EMERGENCY HELPLINES:
• Emergency Police: 100
• Women Helpline (if applicable): 1091 / 181
• Legal Aid Hotline: 15100
• Lawyer Referral Services: [Check state bar council]

═══════════════════════════════════════════════════════════════════════

💡 IMPORTANT LEGAL NOTES:

1. RIGHTS DURING INTERROGATION:
   ✓ You have right to remain silent
   ✓ Police cannot force confession
   ✓ You have right to meet lawyer
   ✓ All statements can be used in court

2. BAIL PRINCIPLES:
   ✓ Bail is normally granted for minor crimes
   ✓ High court can grant bail if district court refuses
   ✓ Bail conditions must be reasonable
   ✓ You can appeal bail decision

3. INVESTIGATION RIGHTS:
   ✓ Request updates on investigation progress
   ✓ Obtain copies of documents filed
   ✓ Challenge illegal police actions
   ✓ File complaint for harassment

4. DOCUMENT PRESERVATION:
   ✓ Keep all FIR related papers safe
   ✓ Get multiple copies of all documents
   ✓ Take screenshots of digital evidence
   ✓ Date and time stamp all records

═══════════════════════════════════════════════════════════════════════

⚖️ DO's AND DON'Ts:

DO:
✓ Be truthful in all statements
✓ Maintain your lawyer relationship
✓ Attend all court hearings
✓ Keep evidence safe
✓ Document harassment
✓ Keep lawyer informed
✓ Follow court orders strictly

DON'T:
✗ Miss any court hearing
✗ Discuss case with accused or their family  
✗ Destroy any evidence
✗ Make statements without lawyer
✗ Try to influence witnesses
✗ Violate bail conditions
✗ Assume guilt without trial

═══════════════════════════════════════════════════════════════════════

📋 DOCUMENTS YOU SHOULD HAVE:

☐ FIR copy (certified)
☐ Police complaint receipt
☐ Bail application (signed by lawyer)
☐ Identity proof
☐ Address proof
☐ Income certificate
☐ Surety documents
☐ Medical reports (if applicable)
☐ Witness statements
☐ Communication records (messages, emails)
☐ This AI-generated legal template (for reference only)

═══════════════════════════════════════════════════════════════════════

⚠️ DISCLAIMER:

These templates and next steps are generated by AI and serve as 
GUIDANCE ONLY. They are NOT a substitute for professional legal advice.

BEFORE FILING ANY DOCUMENT:
• Consult with a qualified criminal lawyer licensed to practice in {normalized['location']}
• Have lawyer review all documents
• Court will require lawyer's signature/representation
• Ensure accuracy of all personal details
• Follow exact court procedures for your jurisdiction

═══════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%d-%m-%Y at %H:%M:%S')}
AI Legal Assistant | Made for Access to Justice (A2J)
"""
        
        return next_steps
