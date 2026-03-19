# Contributing to SYNCCIN 2025 Summer School

Thank you for your interest in contributing to this project! Below you will find the steps to add yourself or someone else as a contributor.

## How to Add a New Contributor

### Step 1: Fork the Repository

1. Go to the [SYNCCIN_2025_summer_school](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school) repository on GitHub.
2. Click the **Fork** button in the top-right corner.
3. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/SYNCCIN_2025_summer_school.git
   cd SYNCCIN_2025_summer_school
   ```

### Step 2: Create a New Branch

```bash
git checkout -b add-contributor/<contributor-name>
```

### Step 3: Add the Contributor to `CONTRIBUTORS.md`

Open or create `CONTRIBUTORS.md` and add an entry for the new contributor in the following format:

```markdown
| Name | Affiliation | Role | GitHub |
|------|-------------|------|--------|
| Full Name | Institution | Role description | [@username](https://github.com/username) |
```

### Step 4: Update the README (optional)

If appropriate, mention the contributor in the `README.md` under the **Contributors** section.

### Step 5: Commit and Push Your Changes

```bash
git add CONTRIBUTORS.md README.md
git commit -m "Add contributor: <contributor-name>"
git push origin add-contributor/<contributor-name>
```

### Step 6: Open a Pull Request

1. Go to your fork on GitHub.
2. Click **Compare & pull request**.
3. Fill in the PR description explaining who is being added and why.
4. Submit the pull request for review.

---

## Contribution Guidelines

- **Notebooks**: Follow the existing notebook structure. Ensure cells run cleanly from top to bottom.
- **Python modules**: Keep functions well-documented with docstrings.
- **Data files**: Do not commit large binary files unless necessary. Prefer cloud-hosted data accessed via URL.
- **Commit messages**: Use clear, descriptive commit messages.

## Code of Conduct

All contributors are expected to be respectful and collaborative. This project is for educational purposes, and contributions that improve learning outcomes are especially welcome.

## Questions?

Open an [issue](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school/issues) or contact the repository author directly.
