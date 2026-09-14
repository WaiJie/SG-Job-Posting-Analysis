# Job Market Pulse
**See where work is moving.**

Job Market Pulse is an experimental analysis of the Singapore job market built from job postings on **MyCareersFuture**.

The project starts with a simple question:

> **What type of job roles are employers actually posting ?**

Answering that question is harder than simply counting job titles.

Job titles are often inconsistent, uncommon, or ambiguous. Employers may use titles such as **Data Analytics Engineer**, **AI Native Engineer**, or other specialised titles that do not map neatly to a standard occupation. The same title can also describe very different work across employers, while different titles can describe highly similar work.

A job posting itself contains much more information: its responsibilities, requirements, qualifications, tools, technologies, and skills.

Job Market Pulse therefore attempts to look beyond the title and **consistently group similar jobs based on the content of their postings**. This creates a structured layer that can be used for deeper analysis.

**Job postings → structured group of roles → skills & tools → market insights**

---

## Why group jobs?

If jobs are analysed only by their titles, the market can become fragmented.

For example, an employer might advertise a **Data Analytics Engineer**, while another uses **Analytics Engineer**, **Data Engineer**, or **AI Native Engineer**. These titles may represent overlapping types of work, but a simple title-based analysis would treat them as separate categories.

The opposite problem also occurs. Two jobs with the same title can involve very different responsibilities.

This makes job titles a poor standalone unit for understanding what employers are actually hiring for.

The goal of Job Market Pulse is therefore to create a **consistent analytical grouping of jobs** based on what the jobs actually contain.

Once that grouping exists, it becomes possible to ask more useful questions:

* What types of work are employers hiring for?
* Which roles are growing or declining?
* What responsibilities are common within each role?
* What tools and technologies are employers asking for?
* Which skills are associated with different types of work?
* How do these patterns differ across industries and job functions?

The role grouping is therefore not the end goal. It is the foundation for the analysis that follows.

---

## Automated role taxonomy

The role taxonomy is built through an automated NLP pipeline rather than relying on job titles alone.

Large language models are used to **extract and normalise information** from each job posting. The **job title, responsibilities, and requirements** are then combined and converted into **semantic embeddings**, which are used for clustering and grouping similar jobs.

Large language models are subsequently used to **label and describe the resulting groups**, turning clusters of similar postings into human-readable role categories.

The overall process is:

**Job posting → LLM extraction → title + responsibilities + requirements → embeddings → clustering → LLM labelling → role groups**

This approach is intended to identify the underlying types of work being advertised while reducing reliance on inconsistent, uncommon, or ambiguous job titles.

The resulting taxonomy is an **analytical grouping**, not an official occupational classification. It is still experimental, and some roles may overlap or be grouped incorrectly. The clustering and labelling methodology will be refined as the project develops.

---

## Job functions

The dashboard uses **43 job-function categories provided by MyCareersFuture**.

These provide a broad, platform-defined view of the market, while the automated role taxonomy provides a more detailed grouping of the jobs within it.

The two layers serve different purposes:

**MyCareersFuture job functions → automated role groups → skills & tools**

The job-function categories therefore reflect how postings are classified on MyCareersFuture and are not intended to represent a universal occupational classification.

---

## What the dashboard shows

The current preview covers **June–August 2026** and provides several views of the job market.

### Market overview

Explore monthly posting activity across the 43 MyCareersFuture job functions, including:

* Total job posts
* New jobs
* Recruiter postings
* Recruiter share
* Monthly changes
* Job-function trends

The dashboard also allows recruiter postings and new-job activity to be viewed separately, since raw posting volume does not necessarily correspond directly to underlying hiring demand.

### Job-function analysis

Compare the volume and movement of different job functions across the observation period.

The current data includes categories such as:

* Engineering
* Building and Construction
* Customer Service
* Information Technology
* Accounting / Auditing / Taxation
* Healthcare / Pharmaceutical
* Banking and Finance
* Logistics / Supply Chain
* Marketing / Public Relations
* and other MyCareersFuture classifications

### Role analysis

Within the broader job functions, explore the automated role groups identified from the job descriptions.

Users can:

* View the most frequently posted roles
* Compare role shares
* Examine month-over-month changes
* Focus on a particular job function
* Inspect examples of postings assigned to a role

This layer is intended to provide a more detailed view of **what kinds of work are actually being advertised**.

---

## From roles to skills and tools

The role taxonomy is the first layer of analysis, not the final output.

Once similar jobs can be grouped consistently, each role can be examined in greater detail:

**Role**
→ responsibilities
→ tools
→ technologies
→ skills
→ qualifications
→ industries
→ employer demand
→ time trends

This allows the project to move beyond:

> **"How many jobs use this title?"**

towards:

> **"What types of work are employers actually hiring for, and what are they asking those workers to do?"**

---

## Market signals

Job Market Pulse distinguishes between several types of posting activity.

**Job posts** represent observed postings during a period.

**New jobs** represent postings identified as new job activity after accounting for repeated or previously observed listings where possible.

**Recruiter posts** identify postings associated with recruitment agencies, allowing users to see how much of the observed market signal comes through recruiters.

For example, a fall in total postings does not necessarily mean that underlying hiring activity has fallen. A market can contain substantial reposting or recruiter activity, which is why the dashboard exposes these signals separately.

---

## Current dataset

**Source:** MyCareersFuture
**Coverage:** June–August 2026
**Geography:** Singapore
**Job functions:** 43

Across all recorded months (June–August 2026):

| Metric               | Jun–Aug 2026 |
| -------------------- | -----------: |
| Total posts observed |      301,156 |
| Unique jobs          |      274,083 |
| Recruiter posts      |      102,022 |
| Recruiter share      |        33.9% |

The figures represent **observable job-posting activity** rather than an estimate of the entire Singapore labour market.

---

## Current release

This is an early preview of the project.

The current release focuses on:

* Market-level posting activity
* Job-function analysis
* Recruiter activity
* New-job activity
* Automated role grouping
* Role-level market analysis

The **role taxonomy is currently the main area for improvement**. Future iterations will test alternative clustering and grouping approaches to improve how well the resulting role groups represent the underlying types of work.

The analysis can then be expanded further into skills, tools, technologies, and other attributes of each role.

---

## Limitations

### Early release

This preview focuses on job-posting activity and role analysis. Additional dimensions may be added in future releases.

### Observed postings, not hires

Posting counts measure observable job-posting activity. They do not directly measure hires, vacancies filled, or employment outcomes.

### Postings are not always unique jobs

Jobs may be reposted, duplicated, or listed in multiple places. Observed posting volume may therefore differ from the number of unique hiring needs.

### Role classification is provisional

Roles are extracted and grouped using local large language models and an experimental taxonomy. Some postings may be assigned to neighbouring or incorrect roles, particularly where occupations share similar skills and responsibilities. **Role-level results may change in future releases.**

### Recruiter postings affect the signal

Recruitment agencies may post on behalf of other employers and can contribute substantially to observed volume. The recruiter filter is provided to make this effect visible.

### Limited observation window

This release covers **June–August 2026** and reflects only observable postings during this period. It should not be interpreted as a complete picture of the Singapore labour market.

---

## Project direction

The long-term goal of Job Market Pulse is to understand the labour market from the **content of jobs**, rather than relying primarily on job titles.

The project starts with a large collection of unstructured job postings and progressively turns them into a structured view of:

**Jobs → Roles → Skills & Tools → Market Insights**

The central challenge is creating a **consistent and useful grouping of jobs**.

Once that foundation is reliable, the same dataset can be used to investigate what different types of work actually involve, which skills and tools employers are asking for, how roles differ across industries, and how demand changes over time.

Ultimately, Job Market Pulse is an attempt to make a large, messy collection of job postings **more legible** — not by assuming that job titles already provide the answer, but by using the information contained within the postings themselves.
