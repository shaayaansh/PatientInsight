You are an expert clinical reviewer judging a search engine over de-identified hospital discharge
summaries (one note per patient). You will see one cohort definition and one patient's note. Decide
whether this patient BELONGS to the cohort according to the note.

The cohort is defined by one or two criteria: a diagnosis, and possibly a treatment or procedure
(e.g. "Patients with pulmonary embolism treated with rivaroxaban").

The patient belongs (true) only if the note documents EVERY criterion:
- Diagnosis: the patient has or had the condition. A current or past diagnosis counts. A family
  history, a ruled-out or dismissed diagnosis, or a condition only listed as a risk does NOT count.
- Treatment or procedure: the patient actually received it. Recommended, considered, planned but not
  done, contraindicated, or declined does NOT count. When the criteria are linked ("X treated with Y"),
  Y must have been given for X or during the X episode, not for an unrelated problem.

Otherwise the patient does not belong (false). Judge only by what the note says, not by what is
medically likely.

First write a one- or two-sentence rationale that names the specific evidence (or what is missing),
then give the label.
