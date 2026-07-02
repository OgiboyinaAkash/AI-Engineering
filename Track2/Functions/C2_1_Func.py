from pathlib import Path
import csv
import json
import math
import re

import numpy as np


def build_corpus_documents():
    """Returns the 60-document multi-domain corpus (id, topic, title, body, query, answer, format)."""
    raw_docs = [
        # ── FINANCE (15) ─────────────────────────────────────────────────────────
        ('finance', 'Loan Amortization Explained',
         'Loan amortization distributes equal monthly payments across the full loan term. '
         'Early payments are weighted heavily toward interest while later payments reduce principal. '
         'A 30-year mortgage at 6 percent APR has a fixed monthly payment calculated from the amortization formula. '
         'Borrowers who make extra principal payments shorten the loan term and reduce total interest paid.',
         'How does loan amortization distribute payments?', 'equal monthly payments'),

        ('finance', 'Portfolio Diversification Theory',
         'Modern portfolio theory shows that diversification reduces unsystematic risk in a portfolio. '
         'Harry Markowitz demonstrated that combining uncorrelated assets lowers overall portfolio volatility. '
         'The efficient frontier represents portfolios with maximum return for each given risk level. '
         'Investors should hold a mix of equities, bonds, and alternatives to optimize the risk-return tradeoff.',
         'What reduces portfolio risk through asset combination?', 'diversification'),

        ('finance', 'Price-to-Earnings Ratio',
         'The price-to-earnings ratio compares a company stock price to its earnings per share. '
         'A high P/E ratio signals that investors expect strong future growth from the company. '
         'Value investors prefer stocks with low P/E ratios relative to industry peers. '
         'The P/E ratio is one of the most widely used equity valuation metrics in fundamental analysis.',
         'What does the P/E ratio compare?', 'stock price to its earnings per share'),

        ('finance', 'Central Bank Interest Rate Policy',
         'Central banks raise interest rates to control inflation by making borrowing more expensive. '
         'Lower interest rates stimulate economic growth by encouraging consumer spending and business investment. '
         'The Federal Reserve uses the federal funds rate as its primary monetary policy tool. '
         'Rate changes ripple through mortgage rates, credit card rates, and bond yields across the economy.',
         'Why do central banks raise interest rates?', 'control inflation'),

        ('finance', 'Bond Duration and Interest Rate Risk',
         'Bond duration measures the sensitivity of a bond price to changes in interest rates. '
         'Longer-duration bonds experience greater price drops when market interest rates rise. '
         'Portfolio managers use duration matching to hedge interest rate risk in fixed-income portfolios. '
         'A bond with a duration of five years loses approximately five percent of value for each one-percent rate increase.',
         'What measures a bond sensitivity to interest rate changes?', 'duration'),

        ('finance', 'Options Contracts Calls and Puts',
         'A call option gives the buyer the right to purchase an asset at the strike price before expiry. '
         'A put option gives the buyer the right to sell an asset at the strike price before expiry. '
         'Options are used for hedging existing positions or speculating on directional price movements. '
         'The Black-Scholes model is the standard framework for pricing European options on non-dividend stocks.',
         'What right does a call option give the buyer?', 'right to purchase'),

        ('finance', 'Credit Scores and Lending Decisions',
         'Credit scores quantify a borrower likelihood of defaulting on a loan repayment. '
         'FICO scores range from 300 to 850 and are calculated from payment history, debt levels, and credit age. '
         'Lenders use credit scores to set loan approval thresholds and determine the applicable interest rate. '
         'A score above 750 typically qualifies borrowers for the best available rates from major lenders.',
         'What do credit scores measure?', 'likelihood of defaulting'),

        ('finance', 'Compound Interest and Long-Term Growth',
         'Compound interest earns returns on both the principal and all previously accumulated interest. '
         'The time value of money states that a dollar today is worth more than a dollar in the future. '
         'An investment of one thousand dollars at seven percent annual compound interest grows to about 7,612 dollars in 30 years. '
         'Starting to invest early dramatically increases the compounding effect over a long horizon.',
         'Why is compound interest powerful for long-term savings?', 'earns returns on principal and accumulated interest'),

        ('finance', 'Inflation and Purchasing Power',
         'Inflation erodes the purchasing power of money held in cash over time. '
         'The Consumer Price Index measures average price changes across a basket of goods and services. '
         'Real investment returns subtract the inflation rate from nominal returns to show actual purchasing power gain. '
         'Investors use Treasury Inflation-Protected Securities to hedge portfolios against sustained inflation risk.',
         'How does inflation affect the value of money?', 'erodes purchasing power'),

        ('finance', 'Financial Statement Analysis',
         'The income statement reports a company revenue, expenses, and net income over a reporting period. '
         'The balance sheet shows assets, liabilities, and shareholders equity at a single point in time. '
         'The cash flow statement tracks cash inflows and outflows from operations, investing, and financing activities. '
         'Analysts use all three statements together to assess a company overall financial health and sustainability.',
         'What does the income statement report?', 'revenue, expenses, and net income'),

        ('finance', 'Index Funds and Passive Investing',
         'Index funds track a market benchmark such as the S&P 500 and hold all constituent securities. '
         'Passive funds have significantly lower expense ratios than actively managed funds. '
         'Decades of data show most actively managed funds underperform their benchmark over ten-year periods. '
         'Low-cost index investing is the standard recommendation for retail investors building long-term wealth.',
         'Why do index funds have lower costs than active funds?', 'lower expense ratios'),

        ('finance', 'Risk and Return in Capital Markets',
         'The risk-return tradeoff states that higher expected returns require the investor to accept higher risk. '
         'Equities historically earn higher returns than bonds but with substantially greater price volatility. '
         'Beta measures a stock sensitivity to market movements relative to a benchmark like the S&P 500. '
         'Investors position themselves on the risk-return spectrum based on their time horizon and risk tolerance.',
         'What does the risk-return tradeoff state?', 'higher returns require accepting higher risk'),

        ('finance', 'Liquidity Risk in Banking',
         'Liquidity risk is an institution inability to meet its short-term financial obligations when due. '
         'Banks maintain pools of liquid assets to fund unexpected deposit withdrawals without fire sales. '
         'The liquidity coverage ratio requires banks to hold high-quality liquid assets for a 30-day stress scenario. '
         'Bank runs occur when depositors lose confidence and simultaneously withdraw funds beyond available liquidity.',
         'What is liquidity risk in banking?', 'inability to meet short-term obligations'),

        ('finance', 'Revenue Recognition Accounting',
         'Revenue recognition determines when a company records revenue in its financial statements. '
         'Under US GAAP revenue is recognized when it is earned and collectible, not simply when cash is received. '
         'The ASC 606 standard requires revenue to be recognized as contractual performance obligations are satisfied. '
         'Misapplied revenue recognition is a frequent source of financial restatements and regulatory enforcement.',
         'When is revenue recognized under accounting standards?', 'performance obligations are satisfied'),

        ('finance', 'Hedge Fund Investment Strategies',
         'Hedge funds use strategies including long-short equity, global macro, and statistical arbitrage. '
         'Long-short equity funds buy undervalued stocks and short overvalued stocks to earn market-neutral returns. '
         'Global macro funds take positions based on broad economic trends across countries and asset classes. '
         'Hedge funds typically charge a two-percent management fee and a twenty-percent performance allocation.',
         'What is the long-short equity strategy?', 'buy undervalued and short overvalued stocks'),

        # ── EDUCATION (15) ───────────────────────────────────────────────────────
        ('education', "Bloom's Taxonomy of Learning Objectives",
         "Bloom's Taxonomy classifies learning objectives into six cognitive levels: remember, understand, apply, analyze, evaluate, and create. "
         'Higher-order thinking skills like analysis and synthesis require deeper engagement than simple recall. '
         "Teachers use Bloom's Taxonomy to design assessments that target the appropriate cognitive demand. "
         'Moving students from memorizing facts to creating original ideas is a hallmark of higher-order instruction.',
         "What are the six levels in Bloom's Taxonomy?", 'remember, understand, apply, analyze, evaluate, and create'),

        ('education', 'Formative vs Summative Assessment',
         'Formative assessments monitor student learning continuously during instruction to inform teaching decisions. '
         'Summative assessments evaluate cumulative student achievement at the end of a unit, semester, or course. '
         'Quizzes, exit tickets, and peer reviews are widely used formative assessment tools in classrooms. '
         'Final exams and standardized tests are summative assessments used to assign grades or certifications.',
         'What is the purpose of formative assessment?', 'monitor student learning during instruction'),

        ('education', 'Zone of Proximal Development',
         "Vygotsky's Zone of Proximal Development describes the gap between what a learner can do alone and with expert guidance. "
         'Effective instruction targets this zone to stretch learners just beyond their current independent ability. '
         'Scaffolding provides temporary structured support that is gradually removed as learner competence grows. '
         'Peer collaboration and targeted teacher feedback are practical tools for supporting learning within this zone.',
         'What does the Zone of Proximal Development describe?', 'gap between what a learner can do alone and with guidance'),

        ('education', 'Active Learning Strategies in the Classroom',
         'Active learning engages students directly in the learning process through structured activities requiring thinking. '
         'Think-pair-share, case studies, and problem-based learning are established active learning approaches. '
         'Research consistently shows active learning produces better retention and transfer than passive lecture. '
         'Flipping the classroom moves direct instruction to pre-class videos, freeing class time for active application.',
         'How does active learning improve student retention?', 'engages students through activities requiring thinking'),

        ('education', 'Universal Design for Learning',
         'Universal Design for Learning provides multiple means of representation, engagement, and expression for all students. '
         'UDL removes learning barriers by building flexibility into curriculum materials and instructional methods. '
         'Offering content in text, audio, and video formats addresses diverse learner preferences and accessibility needs. '
         'UDL shifts responsibility for accessibility from individual accommodations to universal curriculum design.',
         'What three means does Universal Design for Learning provide?', 'representation, engagement, and expression'),

        ('education', 'Metacognition and Self-Regulated Learning',
         'Metacognition is the ability to monitor, evaluate, and regulate one own thinking and learning processes. '
         'Self-regulated learners set goals, select strategies, monitor progress, and reflect on their outcomes. '
         'Teaching metacognitive strategies improves academic performance across subjects and grade levels. '
         'Reflective journaling and structured think-alouds are practical classroom tools for building metacognitive awareness.',
         'What is metacognition in education?', 'monitor and regulate one own thinking'),

        ('education', 'Project-Based Learning',
         'Project-based learning engages students in sustained inquiry around complex, authentic, real-world problems. '
         'Students produce public products or presentations that demonstrate mastery rather than sitting for tests. '
         'PBL develops collaboration, communication, and critical thinking alongside disciplinary content knowledge. '
         'Teachers act as coaches who guide student inquiry rather than delivering information through direct instruction.',
         'What distinguishes project-based learning from traditional instruction?', 'sustained inquiry around real-world problems'),

        ('education', 'Differentiated Instruction',
         'Differentiated instruction tailors teaching to meet the diverse learning needs and readiness levels of students. '
         'Teachers adjust content, process, product, and environment based on ongoing assessment of student profiles. '
         'Tiered assignments allow students at different readiness levels to work toward the same core concept. '
         'Regular formative assessment informs which students need acceleration, enrichment, or additional support.',
         'What does differentiated instruction adjust for each student?', 'content, process, product, and environment'),

        ('education', 'STEM Education and Interdisciplinary Learning',
         'STEM education integrates science, technology, engineering, and mathematics through interdisciplinary problem-solving. '
         'Design challenges like robotics competitions develop technical skills alongside teamwork and communication. '
         'Early exposure to STEM activities increases long-term interest in engineering and computing careers. '
         'Effective STEM lessons connect abstract mathematical concepts to tangible engineering and design problems.',
         'What four subjects does STEM education integrate?', 'science, technology, engineering, and mathematics'),

        ('education', 'Spaced Repetition and Memory Retention',
         'Spaced repetition revisits information at systematically increasing intervals to strengthen long-term memory. '
         'The forgetting curve shows that memory for new material decays rapidly without scheduled review. '
         'Flashcard applications like Anki use spaced repetition algorithms to schedule optimal review sessions automatically. '
         'Combining spaced retrieval practice with interleaving produces the largest documented gains in retention.',
         'How does spaced repetition strengthen memory?', 'revisiting at increasing intervals'),

        ('education', 'Learning Management Systems in Education',
         'Learning Management Systems centralize course materials, assignments, grades, and communication in one platform. '
         'Moodle, Canvas, and Blackboard are widely deployed LMS platforms across K-12 and higher education globally. '
         'Learning analytics derived from LMS data help instructors identify at-risk students before they fail. '
         'Effective LMS adoption requires sustained teacher professional development and clear pedagogical alignment.',
         'What do Learning Management Systems centralize?', 'course materials, assignments, and grades'),

        ('education', 'Rubrics for Transparent Assessment',
         'Rubrics define explicit performance criteria and quality levels used to evaluate student work consistently. '
         'Analytic rubrics score each dimension separately while holistic rubrics assign a single integrated score. '
         'Sharing rubrics with students before an assignment clarifies expectations and measurably improves quality. '
         'Research shows rubrics increase inter-rater reliability and reduce subjective bias in grading.',
         'What do rubrics define to guide assessment?', 'explicit criteria and performance levels'),

        ('education', 'Inquiry-Based Learning',
         'Inquiry-based learning begins with compelling questions or problems and asks students to investigate answers. '
         'Students develop scientific reasoning by forming hypotheses, collecting data, and drawing evidence-based conclusions. '
         'The teacher role shifts from information deliverer to questioning facilitator who scaffolds the inquiry process. '
         'Inquiry learning is most effective when students possess sufficient background knowledge to engage the problem.',
         'How does inquiry-based learning start?', 'starts with questions and students investigate'),

        ('education', 'Effective Feedback Practices',
         'Effective feedback is timely, specific, and actionable, and focuses on the task rather than the learner. '
         "John Hattie's meta-analysis identified feedback as one of the most powerful influences on student achievement. "
         'Delayed or vague feedback has negligible effect on improving subsequent student performance. '
         'Structured peer feedback, when properly scaffolded, can be as effective as teacher feedback for skill development.',
         'What qualities make feedback effective in education?', 'timely, specific, and actionable'),

        ('education', 'Curriculum Alignment and Backward Design',
         'Curriculum alignment ensures learning objectives, instructional activities, and assessments form a coherent system. '
         'Backward design begins with desired outcomes and plans instruction and assessments to achieve those outcomes. '
         'Misaligned curricula produce students who study material they are never assessed on or vice versa. '
         'Regular curriculum review keeps learning objectives relevant to evolving workforce and societal requirements.',
         'What does curriculum alignment ensure?', 'objectives, instruction, and assessments are coherently linked'),

        # ── HEALTHCARE (10) ──────────────────────────────────────────────────────
        ('healthcare', 'Randomized Controlled Trials',
         'Randomized controlled trials are the gold standard for evaluating new medical treatments. '
         'Participants are randomly assigned to treatment or placebo groups to minimize selection and confounding bias. '
         'Blinding prevents participants and researchers from knowing group assignment, reducing measurement bias. '
         'Systematic reviews synthesize evidence across multiple trials to produce the strongest clinical guidance.',
         'What is the gold standard for evaluating medical treatments?', 'randomized controlled trials'),

        ('healthcare', 'Electronic Health Records',
         'Electronic Health Records store patient medical histories, diagnoses, medications, and treatment plans digitally. '
         'EHR systems improve care coordination by making records accessible across providers and care settings. '
         'Interoperability standards such as HL7 FHIR allow different EHR systems to exchange patient data securely. '
         'Poorly designed EHR interfaces increase clinician cognitive burden and contribute to professional burnout.',
         'What do Electronic Health Records store for patients?', 'medical histories, diagnoses, and treatment plans'),

        ('healthcare', 'Medication Adherence in Chronic Disease',
         'Medication adherence means patients take prescribed drugs in the correct dose at the correct time. '
         'Non-adherence accounts for approximately half of all treatment failures in patients with chronic conditions. '
         'Simplified dosing regimens, patient education, and reminder applications reliably improve adherence rates. '
         'Poor adherence costs the US healthcare system an estimated 300 billion dollars annually in avoidable costs.',
         'What is medication adherence?', 'taking prescribed drugs as directed'),

        ('healthcare', 'Preventive Care and Early Screening',
         'Preventive care reduces disease incidence and severity through early detection and targeted intervention. '
         'Screenings for cancer, diabetes, and cardiovascular disease identify conditions before symptoms develop. '
         'Vaccination prevents infectious diseases and builds community immunity through population-level herd protection. '
         'Investment in preventive care consistently delivers a higher return than treating advanced disease.',
         'How does preventive care reduce disease burden?', 'early detection and intervention'),

        ('healthcare', 'Telehealth and Remote Patient Monitoring',
         'Telehealth delivers healthcare services through video consultations, phone calls, and digital health platforms. '
         'Remote monitoring devices track vital signs such as blood pressure and blood glucose levels at home. '
         'Telehealth adoption accelerated during the COVID-19 pandemic to maintain continuity of care safely. '
         'Clinical evidence shows telehealth achieves equivalent outcomes to in-person care for many primary care conditions.',
         'What does telehealth deliver?', 'healthcare services through video and digital platforms'),

        ('healthcare', 'Patient Safety and Error Prevention',
         'Medical errors remain a leading cause of preventable patient harm and death in hospitals worldwide. '
         'Surgical checklists, standardized protocols, and simulation training measurably reduce clinical error rates. '
         'Root cause analysis investigates systemic and process failures rather than assigning individual blame. '
         'A just culture encourages staff to report near-misses and adverse events without fear of punitive response.',
         'How do hospitals reduce preventable medical errors?', 'checklists and standardized protocols'),

        ('healthcare', 'Cognitive Behavioral Therapy',
         'Cognitive Behavioral Therapy treats mental health disorders by identifying and changing negative thought patterns. '
         'CBT is an evidence-based treatment for depression, anxiety disorders, and post-traumatic stress disorder. '
         'Patients learn to recognize cognitive distortions and replace them with more accurate, balanced perspectives. '
         'CBT is typically delivered in 12 to 20 structured sessions with explicit goal-setting and outcome tracking.',
         'What does Cognitive Behavioral Therapy change?', 'negative thought patterns'),

        ('healthcare', 'Chronic Disease Management',
         'Chronic diseases such as diabetes, hypertension, and heart failure require coordinated long-term management. '
         'Self-management education empowers patients to actively monitor symptoms and adjust their own behavior. '
         'Multidisciplinary care teams coordinate treatment across physicians, nurses, dietitians, and pharmacists. '
         'Patient registries and population health analytics identify high-risk individuals who need proactive outreach.',
         'What approach is most effective for managing chronic diseases?', 'multidisciplinary care teams'),

        ('healthcare', 'Medical Imaging and Diagnostics',
         'Medical imaging modalities include X-ray, computed tomography, magnetic resonance imaging, and ultrasound. '
         'AI algorithms trained on large imaging datasets can detect tumors and anomalies with radiologist-level accuracy. '
         'MRI uses magnetic fields and radio waves and does not expose patients to ionizing radiation unlike CT and X-ray. '
         'Early imaging diagnosis enables treatment before conditions become life-threatening or surgically complex.',
         'What imaging technique avoids ionizing radiation?', 'MRI'),

        ('healthcare', 'Antibiotic Resistance',
         'Antibiotics treat bacterial infections by killing bacteria or inhibiting their reproduction. '
         'Antibiotic resistance develops when bacteria evolve mechanisms that allow them to survive antibiotic exposure. '
         'Overuse, inappropriate prescribing, and agricultural antibiotic use are the primary drivers of resistance. '
         'The World Health Organization classifies antibiotic resistance as one of the greatest global public health threats.',
         'What causes antibiotic resistance to develop?', 'overuse and inappropriate prescribing'),

        # ── TECHNOLOGY (10) ──────────────────────────────────────────────────────
        ('technology', 'Cloud Computing Service Models',
         'Cloud computing delivers on-demand computing resources over the internet using a pay-as-you-go pricing model. '
         'Infrastructure as a Service provides virtualized compute, storage, and networking managed by the cloud provider. '
         'Platform as a Service offers a managed runtime and development tools for building and deploying applications. '
         'Software as a Service delivers fully managed applications such as email, CRM, and collaboration tools over the web.',
         'What are the three cloud service delivery models?', 'IaaS, PaaS, and SaaS'),

        ('technology', 'REST API Design Principles',
         'RESTful APIs use standard HTTP methods including GET, POST, PUT, and DELETE to operate on named resources. '
         'Resources are identified by URIs and represented in standard formats such as JSON or XML. '
         'Statelessness requires each API request to carry all information needed without relying on server-side session state. '
         'API versioning with path prefixes such as slash v1 allows backward-compatible evolution of the interface.',
         'What HTTP methods do REST APIs use?', 'GET, POST, PUT, and DELETE'),

        ('technology', 'CI/CD and DevOps Pipelines',
         'DevOps integrates software development and IT operations practices to shorten the delivery cycle. '
         'Continuous Integration automatically builds and tests code changes when developers push to a shared repository. '
         'Continuous Delivery extends CI by automatically deploying tested code to staging or production environments. '
         'Automated pipelines reduce human error and provide rapid feedback between code commit and running software.',
         'What does Continuous Integration automatically do?', 'builds and tests code changes'),

        ('technology', 'Cybersecurity and Zero Trust',
         'Zero trust security assumes no user, device, or network segment is trusted by default. '
         'All access requests are verified continuously using identity, device health, and contextual signals. '
         'Least privilege principles restrict access rights to the minimum needed to perform a given role. '
         'The STRIDE framework categorizes threats as spoofing, tampering, repudiation, information disclosure, denial of service, and elevation of privilege.',
         'What does zero trust security assume?', 'no user or device is trusted by default'),

        ('technology', 'Software Testing Levels',
         'Unit tests verify the behavior of individual functions or components in complete isolation from dependencies. '
         'Integration tests check that multiple components interact correctly when wired together. '
         'End-to-end tests simulate complete user workflows through the full application stack. '
         'Test-driven development writes failing tests before code, using test failures to guide implementation decisions.',
         'What do unit tests verify?', 'individual functions in isolation'),

        ('technology', 'Containerization with Docker',
         'Containers package an application and all its runtime dependencies into a portable, isolated execution unit. '
         'Docker is the dominant container platform for building, distributing, and running containerized workloads. '
         'Kubernetes orchestrates containers at scale across clusters, handling scheduling, scaling, and self-healing. '
         'Containers enforce consistency between development, testing, and production deployment environments.',
         'What do containers package for portability?', 'application and its dependencies'),

        ('technology', 'Database Indexing',
         'Database indexes speed up query execution by enabling the engine to locate rows without a full table scan. '
         'B-tree indexes are the default index structure and efficiently support both equality and range queries. '
         'Adding too many indexes degrades write performance because every index must be updated on each insert or update. '
         'Composite indexes covering multiple columns reduce the number of index reads for multi-column filter queries.',
         'How do database indexes improve query speed?', 'locate rows without scanning the full table'),

        ('technology', 'Microservices Architecture',
         'Microservices decompose a monolithic application into small, independently deployable services. '
         'Each service owns its own data store and communicates with peers through APIs or asynchronous message queues. '
         'Independent deployability allows teams to release individual services without coordinating full-system releases. '
         'Distributed systems introduce operational complexity including latency, partial failures, and distributed tracing needs.',
         'How do microservices improve deployment flexibility?', 'independently deployable services'),

        ('technology', 'Encryption and Data Protection',
         'Encryption converts plaintext data into ciphertext that is unreadable without the correct decryption key. '
         'AES-256 is the industry standard symmetric encryption algorithm for protecting data at rest. '
         'TLS encrypts data in transit between clients and servers to prevent interception and tampering. '
         'Public-key infrastructure enables secure key exchange over untrusted networks without sharing private keys.',
         'What does encryption protect data from?', 'interception and unauthorized access'),

        ('technology', 'Version Control with Git',
         'Git is a distributed version control system that tracks every change made to source code over time. '
         'Feature branches allow developers to work in isolation without disrupting the main production codebase. '
         'Pull requests enable structured code review and discussion before merging changes into the main branch. '
         "Git's commit history forms a permanent audit trail showing what changed, when it changed, and the reason why.",
         'What does Git track for source code projects?', 'every change to source code over time'),

        # ── AI / ML (5) ──────────────────────────────────────────────────────────
        ('ai', 'Backpropagation and Gradient Descent',
         'Backpropagation computes gradients through a neural network by propagating prediction error backward. '
         'Gradient descent updates model weights iteratively in the direction that minimizes the loss function. '
         'Learning rate controls step size; too large causes training divergence, too small causes slow convergence. '
         'The Adam optimizer adapts per-parameter learning rates using first and second gradient moment estimates.',
         'What computes gradients for neural network training?', 'backpropagation'),

        ('ai', 'Transformer Architecture and Self-Attention',
         'The Transformer architecture replaces recurrence with self-attention mechanisms to model token relationships. '
         'Attention weights determine how much each token influences the contextual representation of every other token. '
         'BERT and GPT are large pre-trained Transformer models fine-tuned on downstream natural language tasks. '
         'Scaling Transformer model size and training data increases capability but requires exponentially more compute.',
         'What mechanism does the Transformer architecture use?', 'self-attention'),

        ('ai', 'Overfitting and Regularization Techniques',
         'Overfitting occurs when a model memorizes training data noise and fails to generalize to unseen examples. '
         'Dropout randomly deactivates neurons during training to prevent co-adaptation of feature detectors. '
         'L2 weight decay penalizes large parameter values by adding their squared magnitude to the loss function. '
         'Early stopping halts training when validation loss stops improving, preserving the best generalization checkpoint.',
         'What technique prevents overfitting by deactivating neurons?', 'dropout'),

        ('ai', 'RAG vs Fine-Tuning for Production LLMs',
         'Retrieval-Augmented Generation grounds LLM responses in retrieved documents to reduce hallucination. '
         'Fine-tuning adjusts model weights on labeled task-specific data to modify behavior, tone, or reasoning style. '
         'RAG is preferred for knowledge-intensive tasks because the index can be updated without any model retraining. '
         'Fine-tuning is preferred for consistent output format and specialized domain terminology not present in pretraining.',
         'When should you prefer RAG over fine-tuning?', 'dynamic knowledge bases'),

        ('ai', 'Embeddings and Vector Search',
         'Embeddings are dense fixed-length vectors that encode the semantic meaning of text in continuous space. '
         'Semantically similar texts produce embedding vectors with high cosine similarity. '
         'Vector databases such as Pinecone and Weaviate store embeddings and support approximate nearest neighbor search. '
         'Approximate nearest neighbor algorithms enable millisecond-latency retrieval across millions of vectors.',
         'What do text embeddings encode?', 'semantic meaning of text'),

        # ── LEGAL (5) ────────────────────────────────────────────────────────────
        ('legal', 'Contract Formation Requirements',
         'A valid contract requires offer, acceptance, consideration, and mutual assent between competent parties. '
         'Consideration is something of value exchanged by each party that makes the agreement legally enforceable. '
         'Contracts can be voided if formed through duress, fraud, undue influence, or material misrepresentation. '
         'Written contracts are preferred over oral agreements because they provide clear evidentiary records.',
         'What four elements are required to form a valid contract?', 'offer, acceptance, consideration, and mutual assent'),

        ('legal', 'Intellectual Property Law Overview',
         'Intellectual property law protects creations of the mind including inventions, brands, and creative works. '
         'Patents grant inventors exclusive rights to their inventions for up to 20 years from the filing date. '
         'Trademarks protect distinctive brand identifiers such as logos, names, and slogans used in commerce. '
         'Copyright arises automatically upon creation of an original work and protects expression, not ideas.',
         'What does copyright protect?', 'original creative works'),

        ('legal', 'GDPR Data Privacy Requirements',
         'The General Data Protection Regulation requires organizations to obtain explicit consent before collecting personal data. '
         'Data subjects have legal rights to access, correct, and request deletion of their personal data. '
         'Organizations experiencing a data breach must notify supervisory authorities within 72 hours of discovery. '
         'Non-compliance penalties can reach four percent of annual global turnover or 20 million euros, whichever is higher.',
         'What rights do individuals have under GDPR?', 'access, correct, and delete their data'),

        ('legal', 'Employment Law Fundamentals',
         'Employment law governs the legal relationship between employers and their employees. '
         'At-will employment allows either party to end the employment relationship at any time without cause. '
         'Anti-discrimination statutes prohibit adverse employment actions based on protected characteristics. '
         'Minimum wage laws establish a compensation floor that employers must honor regardless of any agreement.',
         'What does employment law govern?', 'relationship between employers and employees'),

        ('legal', 'Corporate Compliance Programs',
         'Compliance programs help organizations identify, prevent, and correct violations of laws and regulations. '
         'Effective programs include a written code of conduct, risk assessments, employee training, and monitoring. '
         'Leadership commitment and anonymous reporting channels are essential for building a culture of compliance. '
         'External regulatory audits assess whether compliance controls are adequately designed and actually operating.',
         'What do corporate compliance programs help organizations prevent?', 'violations of laws and regulations'),
    ]

    documents = []
    for i, (topic, title, body, query, answer) in enumerate(raw_docs, start=1):
        documents.append({
            'id': i, 'topic': topic, 'title': title,
            'body': body, 'query': query, 'answer': answer
        })

    # Assign formats cyclically: first 15 -> txt, next 15 -> md, next 15 -> jsonl, last 15 -> csv
    formats = ['txt'] * 15 + ['md'] * 15 + ['jsonl'] * 15 + ['csv'] * 15
    for doc, fmt in zip(documents, formats):
        doc['format'] = fmt

    return documents


def persist_corpus(documents, data_dir):
    """
    Writes the corpus to disk (txt/md/jsonl/csv) plus an eval-queries manifest.
    Idempotent: skips the (slow) content writes if already persisted, but
    always refreshes the small eval-queries manifest.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / 'docs.jsonl').exists():
        for doc in documents:
            if doc['format'] == 'txt':
                path = data_dir / f"doc_{doc['id']:03d}.txt"
                path.write_text(f"Title: {doc['title']}\n{doc['body']}\n", encoding='utf-8')
            elif doc['format'] == 'md':
                path = data_dir / f"doc_{doc['id']:03d}.md"
                path.write_text(f"# {doc['title']}\n\n{doc['body']}\n", encoding='utf-8')

        with (data_dir / 'docs.jsonl').open('w', encoding='utf-8') as f:
            for doc in documents:
                if doc['format'] == 'jsonl':
                    f.write(json.dumps({'id': doc['id'], 'title': doc['title'], 'body': doc['body']}) + '\n')

        with (data_dir / 'docs.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'title', 'body'])
            writer.writeheader()
            for doc in documents:
                if doc['format'] == 'csv':
                    writer.writerow({'id': doc['id'], 'title': doc['title'], 'body': doc['body']})

    eval_doc_ids = {4, 7, 9, 16, 17, 20, 31, 33, 36, 41, 43, 48, 51, 53, 56, 58}
    eval_rows = [
        {'query': doc['query'], 'doc_id': doc['id'], 'answer': doc['answer']}
        for doc in documents if doc['id'] in eval_doc_ids
    ]
    # .json (not .jsonl) so it is not picked up by load_corpus's *.jsonl glob
    with (data_dir / 'eval_queries.json').open('w', encoding='utf-8') as f:
        json.dump(eval_rows, f)


def load_eval_queries(data_dir):
    """Loads the persisted evaluation query set: list of {query, doc_id, answer}."""
    with (data_dir / 'eval_queries.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def load_txt(path):
    text = path.read_text(encoding='utf-8')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0].replace('Title: ', '') if lines else 'Untitled'
    body = ' '.join(lines[1:]) if len(lines) > 1 else ''
    return title, body


def load_md(path):
    text = path.read_text(encoding='utf-8')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0].replace('# ', '') if lines else 'Untitled'
    body = ' '.join(lines[1:]) if len(lines) > 1 else ''
    return title, body


def load_jsonl(path):
    docs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append({
                'id': int(item['id']),
                'title': item['title'],
                'body': item['body'],
                'source': 'jsonl'
            })
    return docs


def load_csv(path):
    docs = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({
                'id': int(row['id']),
                'title': row['title'],
                'body': row['body'],
                'source': 'csv'
            })
    return docs


def load_corpus(data_dir):
    docs = []
    for path in data_dir.glob('*.txt'):
        title, body = load_txt(path)
        docs.append({
            'id': int(path.stem.split('_')[1]),
            'title': title,
            'body': body,
            'source': 'txt'
        })
    for path in data_dir.glob('*.md'):
        title, body = load_md(path)
        docs.append({
            'id': int(path.stem.split('_')[1]),
            'title': title,
            'body': body,
            'source': 'md'
        })
    for path in data_dir.glob('*.jsonl'):
        docs.extend(load_jsonl(path))
    for path in data_dir.glob('*.csv'):
        docs.extend(load_csv(path))
    docs.sort(key=lambda d: d['id'])
    return docs


def build_comparison_corpus():
    """Returns the 6-document chunking-comparison corpus plus its eval queries."""
    documents = [
        {
            'id': 301,
            'title': 'Central Bank Policy and Bond Duration',
            'body': (
                'Central banks raise interest rates to control inflation by making borrowing more expensive. '
                'When rates rise, existing bond prices fall because newer bonds offer higher yields. '
                'Bond duration measures the sensitivity of a bond price to changes in interest rates. '
                'A portfolio manager uses duration matching to hedge against rising rate environments.'
            )
        },
        {
            'id': 302,
            'title': 'Credit Risk and Portfolio Diversification',
            'body': (
                'Credit scores quantify a borrower likelihood of defaulting on a loan repayment. '
                'Lenders combine credit scores with income verification to set approval thresholds. '
                'Diversification reduces unsystematic risk by spreading exposure across uncorrelated assets. '
                'A well-diversified portfolio limits the impact of any single credit default on total returns.'
            )
        },
        {
            'id': 303,
            'title': "Bloom's Taxonomy and Formative Assessment",
            'body': (
                "Bloom's Taxonomy classifies learning objectives into six cognitive levels from remember to create. "
                'Higher-order objectives like analysis and evaluation require more demanding instructional design. '
                'Formative assessments monitor student learning continuously during instruction to inform teaching. '
                'Exit tickets aligned to specific Bloom levels give teachers actionable data within a single lesson.'
            )
        },
        {
            'id': 304,
            'title': 'Spaced Repetition and Active Learning',
            'body': (
                'Spaced repetition revisits information at increasing intervals to strengthen long-term memory. '
                'The forgetting curve shows that memory decays rapidly without scheduled review sessions. '
                'Active learning engages students through structured activities rather than passive listening. '
                'Combining spaced review with active retrieval practice produces the largest gains in retention.'
            )
        },
        {
            'id': 305,
            'title': 'EHR Systems and Medication Adherence',
            'body': (
                'Electronic Health Records store patient medical histories, diagnoses, and treatment plans digitally. '
                'EHR alerts can notify clinicians when prescribed medications have dangerous interactions. '
                'Medication adherence means patients take prescribed drugs in the correct dose at the correct time. '
                'Reminder features in EHR patient portals measurably improve adherence rates for chronic conditions.'
            )
        },
        {
            'id': 306,
            'title': 'Microservices and Containerization',
            'body': (
                'Microservices decompose a monolithic application into small, independently deployable services. '
                'Each service owns its own data store and exposes a well-defined API to other services. '
                'Containers package an application and its runtime dependencies into a portable, isolated unit. '
                'Kubernetes orchestrates containers across clusters and handles scaling and self-healing automatically.'
            )
        },
    ]

    queries = [
        {'query': 'Why do central banks raise interest rates?', 'answer': 'control inflation', 'doc_id': 301},
        {'query': 'What measures bond price sensitivity to rate changes?', 'answer': 'duration', 'doc_id': 301},
        {'query': "What does Bloom's Taxonomy classify?", 'answer': 'learning objectives', 'doc_id': 303},
        {'query': 'How does spaced repetition improve memory?', 'answer': 'revisits information at increasing intervals', 'doc_id': 304},
        {'query': 'What do Electronic Health Records store?', 'answer': 'medical histories, diagnoses', 'doc_id': 305},
        {'query': 'How do microservices decompose applications?', 'answer': 'independently deployable services', 'doc_id': 306},
    ]

    return documents, queries


def persist_comparison_corpus(documents, queries, corpus_dir):
    """Idempotent: writes the comparison corpus + queries if not already persisted."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / 'docs.jsonl'
    if not path.exists():
        with path.open('w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        with (corpus_dir / 'queries.json').open('w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2)
        return True
    return False


def load_comparison_corpus(corpus_dir):
    """Reads the persisted comparison corpus + queries back from disk."""
    documents = []
    with (corpus_dir / 'docs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    with (corpus_dir / 'queries.json').open(encoding='utf-8') as f:
        queries = json.load(f)
    return documents, queries


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = None

    def fit(self, texts):
        doc_count = len(texts)
        df = {}
        for text in texts:
            terms = set(tokenize(text))
            for term in terms:
                df[term] = df.get(term, 0) + 1
        self.vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + doc_count) / (1 + df[term])) + 1.0
        return self

    def transform(self, texts):
        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        if not self.vocab:
            return vectors
        for i, text in enumerate(texts):
            terms = tokenize(text)
            if not terms:
                continue
            counts = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            for term, count in counts.items():
                idx = self.vocab.get(term)
                if idx is None:
                    continue
                tf = count / len(terms)
                vectors[i, idx] = tf * self.idf[idx]
        return vectors

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)


def fixed_size_chunking(text, chunk_size=40, overlap=5):
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
    return chunks


def sentence_aware_chunking(text, max_words=60, overlap_sentences=1):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap = current_chunk[-overlap_sentences:]
            current_chunk = overlap.copy()
            current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += sentence_len
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def cosine_similarity(vec_a, vec_b):
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-10
    return float(np.dot(vec_a, vec_b) / denom)


def semantic_chunking(text, similarity_threshold=0.25, min_sentences=2, max_words=80):
    sentences = split_sentences(text)
    if not sentences:
        return []
    vectorizer = SimpleTfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0].split())
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(vectors[i - 1], vectors[i])
        sentence_len = len(sentences[i].split())
        should_append = (
            similarity >= similarity_threshold
            or len(current_chunk) < min_sentences
        )
        if should_append and current_len + sentence_len <= max_words:
            current_chunk.append(sentences[i])
            current_len += sentence_len
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_len = sentence_len
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def build_chunks(docs, chunk_fn):
    chunks = []
    for doc in docs:
        text = f"{doc['title']}. {doc['body']}"
        for chunk in chunk_fn(text):
            chunks.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': chunk
            })
    return chunks


def cosine_similarity_scores(query_vec, doc_vecs):
    query_norm = np.linalg.norm(query_vec) + 1e-10
    doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-10
    return (doc_vecs @ query_vec) / (doc_norms * query_norm)


def retrieve_top_k(query, chunks, vectorizer, chunk_vectors, top_k=3):
    query_vec = vectorizer.transform([query])[0]
    scores = cosine_similarity_scores(query_vec, chunk_vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def evaluate_chunking(method_name, chunk_fn, docs, queries, top_k=3):
    chunks = build_chunks(docs, chunk_fn)
    avg_len = float(np.mean([len(c['text'].split()) for c in chunks]))
    vectorizer = SimpleTfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform([c['text'] for c in chunks])
    top1 = 0
    top3 = 0
    mrr_scores = []
    for item in queries:
        retrieved = retrieve_top_k(item['query'], chunks, vectorizer, chunk_vectors, top_k=top_k)
        answer = item['answer'].lower()
        found_rank = None
        for idx, chunk in enumerate(retrieved):
            if answer in chunk['text'].lower():
                found_rank = idx + 1
                break
        if found_rank == 1:
            top1 += 1
        if found_rank is not None and found_rank <= top_k:
            top3 += 1
        mrr_scores.append(1.0 / found_rank if found_rank else 0.0)
    total = len(queries)
    return {
        'method': method_name,
        'chunks': len(chunks),
        'avg_len': avg_len,
        'top1': top1 / total if total else 0.0,
        'top3': top3 / total if total else 0.0,
        'mrr': float(np.mean(mrr_scores)) if mrr_scores else 0.0
    }


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1       # term frequency saturation parameter
        self.b  = b        # document length normalisation parameter
        self._corpus = []
        self._df     = {}
        self._idf    = {}
        self._avgdl  = 0
        self._N      = 0

    def fit(self, texts):
        self._corpus = [tokenize(t) for t in texts]
        self._N      = len(self._corpus)
        self._avgdl  = sum(len(d) for d in self._corpus) / max(self._N, 1)
        df = {}
        for doc in self._corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self._df[term] = freq
            self._idf[term] = math.log(
                (self._N - freq + 0.5) / (freq + 0.5) + 1.0
            )
        return self

    def _score(self, query_terms, doc_idx):
        doc  = self._corpus[doc_idx]
        dl   = len(doc)
        tf_map = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in self._idf:
                continue
            tf   = tf_map.get(term, 0)
            num  = tf * (self.k1 + 1)
            den  = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += self._idf[term] * (num / den)
        return score

    def retrieve(self, query, top_k=5):
        qt     = tokenize(query)
        scores = [self._score(qt, i) for i in range(self._N)]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]


def bm25_retrieve(query, bm25_instance, chunks, top_k=3):
    indices = bm25_instance.retrieve(query, top_k=top_k)
    return [chunks[i] for i in indices]


def retrieve_with_threshold(query, chunks, vectorizer, chunk_vectors, threshold=0.20, max_k=5):
    query_vec = vectorizer.transform([query])[0]
    scores = cosine_similarity_scores(query_vec, chunk_vectors)
    filtered = [(i, s) for i, s in enumerate(scores) if s >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:max_k]
    return [chunks[i] for i, _ in filtered]


def eval_threshold(threshold, queries, chunks, vectorizer, chunk_vectors, max_k=5):
    hits = 0
    retrieved_counts = []
    for item in queries:
        retrieved = retrieve_with_threshold(
            item['query'], chunks, vectorizer, chunk_vectors,
            threshold=threshold, max_k=max_k
        )
        retrieved_counts.append(len(retrieved))
        if any(item['answer'].lower() in c['text'].lower() for c in retrieved):
            hits += 1
    hit_rate = hits / len(queries)
    avg_returned = float(np.mean(retrieved_counts))
    return hit_rate, avg_returned


def eval_top_k(k, queries, chunks, vectorizer, chunk_vectors):
    hits = 0
    retrieved_counts = []
    for item in queries:
        retrieved = retrieve_top_k(item['query'], chunks, vectorizer, chunk_vectors, top_k=k)
        retrieved_counts.append(len(retrieved))
        if any(item['answer'].lower() in c['text'].lower() for c in retrieved):
            hits += 1
    hit_rate = hits / len(queries)
    avg_returned = float(np.mean(retrieved_counts))
    return hit_rate, avg_returned


def eval_retrieval_method(method_name, retrieve_fn, queries, top_k=3):
    top1, top3, mrr_scores = 0, 0, []
    for item in queries:
        results = retrieve_fn(item['query'], top_k)
        answer  = item['answer'].lower()
        rank    = None
        for pos, chunk_or_doc in enumerate(results):
            text = (chunk_or_doc.get('text', '') or chunk_or_doc.get('body', '')).lower()
            if answer in text:
                rank = pos + 1
                break
        if rank == 1:           top1 += 1
        if rank and rank <= 3:  top3 += 1
        mrr_scores.append(1.0 / rank if rank else 0.0)
    total = len(queries)
    return {
        'method': method_name,
        'top1':   top1 / total if total else 0.0,
        'top3':   top3 / total if total else 0.0,
        'mrr':    float(np.mean(mrr_scores)) if mrr_scores else 0.0,
    }


def lexical_search(terms, docs, phrase=None, strict_and=False, scope='all'):
    scores = {}
    for doc_id, content in docs.items():
        title = content['title'].lower()
        body = content['body'].lower()
        if scope == 'title':
            searchable = title
        elif scope == 'body':
            searchable = body
        else:
            searchable = title + ' ' + body
        matched = 0
        for term in terms:
            if term in searchable:
                matched += 1
                scores[doc_id] = scores.get(doc_id, 0) + 1
        if strict_and and matched < len(terms):
            scores.pop(doc_id, None)
            continue
        if phrase and phrase in searchable:
            scores[doc_id] = scores.get(doc_id, 0) + 5
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


def run_retrieval(query_text, docs, opts=None):
    if opts is None:
        terms = tokenize(query_text)
        phrase = None
        strict_and = False
        scope = 'all'
    else:
        terms = opts.get('terms', tokenize(query_text))
        phrase = opts.get('phrase')
        strict_and = opts.get('strict_and', False)
        scope = opts.get('scope', 'all')
    return lexical_search(terms, docs, phrase=phrase, strict_and=strict_and, scope=scope)


def rewrite_expansion(query_text, synonyms):
    terms = tokenize(query_text)
    expanded = []
    for term in terms:
        expanded.append(term)
        for syn in synonyms.get(term, []):
            expanded.extend(tokenize(syn))
    return {'terms': expanded, 'strict_and': False}


def rewrite_relaxation(query_text, relax_terms):
    terms = [t for t in tokenize(query_text) if t not in relax_terms]
    return {'terms': terms, 'strict_and': False}


def rewrite_segmentation(query_text, phrases):
    phrase = None
    lowered = query_text.lower()
    for p in phrases:
        if p in lowered:
            phrase = p
            break
    return {'terms': tokenize(query_text), 'phrase': phrase, 'strict_and': False}


def rewrite_scoping(query_text):
    return {'terms': tokenize(query_text), 'scope': 'title', 'strict_and': False}


def evaluate_lexical(queries, docs, opts_fn=None):
    top1 = 0
    top3 = 0
    for item in queries:
        opts = opts_fn(item['query']) if opts_fn else None
        results = run_retrieval(item['query'], docs, opts)
        if results and results[0] == item['doc_id']:
            top1 += 1
        if item['doc_id'] in results[:3]:
            top3 += 1
    total = len(queries)
    return {
        'top1': top1 / total if total else 0.0,
        'top3': top3 / total if total else 0.0
    }
