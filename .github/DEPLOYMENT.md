# Deployment Guide

This document describes how to deploy diwasp-python to PyPI and GitHub Pages.

## PyPI Deployment

### Prerequisites

1. **Set up PyPI Trusted Publishing** (recommended):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher for your GitHub repository
   - Repository: `wavespectra/diwasp-python`
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi`

2. **Alternative: API Token** (if not using trusted publishing):
   - Generate a PyPI API token at https://pypi.org/manage/account/token/
   - Add it as a GitHub secret named `PYPI_API_TOKEN`
   - Update the workflow to use the token instead of trusted publishing

### Publishing Process

The package is automatically published to PyPI when you create a new release:

1. **Update version** in `pyproject.toml`:

   ```toml
   version = "0.2.0"
   ```

2. **Create a git tag**:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

3. **Create a GitHub Release**:
   - Go to https://github.com/wavespectra/diwasp-python/releases/new
   - Select the tag you just created
   - Add release notes describing changes
   - Click "Publish release"

4. **Automatic deployment**:
   - The `publish-pypi.yml` workflow will automatically trigger
   - It will build the package and publish to PyPI
   - Check the Actions tab for progress

### Testing on TestPyPI

To test the publishing process without affecting the production PyPI:

1. Set up TestPyPI trusted publishing at https://test.pypi.org/manage/account/publishing/

2. Manually trigger the workflow:
   - Go to Actions → Publish to PyPI
   - Click "Run workflow"
   - This will publish to TestPyPI only

3. Test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ diwasp
   ```

## GitHub Pages Deployment

### Initial Setup

1. **Enable GitHub Pages**:
   - Go to your GitHub repository Settings → Pages
   - Under "Build and deployment", set Source to "GitHub Actions"
   - No branch selection needed - the workflow handles deployment

2. **Verify workflow permissions**:
   - Go to Settings → Actions → General
   - Under "Workflow permissions", ensure "Read and write permissions" is selected
   - Check "Allow GitHub Actions to create and approve pull requests" if needed

### Building Documentation

Documentation is built and deployed automatically:

- **On every push to main/master**: Full documentation rebuild and deployment
- **On pull requests**: Build only (no deployment) to verify docs build correctly
- **Manual trigger**: Use the workflow dispatch button in Actions tab

### Accessing Documentation

Once deployed, documentation will be available at:

- https://wavespectra.github.io/diwasp-python/

### Local Documentation Build

To build documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Continuous Integration

### Test Workflow

The `tests.yml` workflow runs automatically on:

- Push to main/master/develop branches
- Pull requests to main/master/develop
- Manual trigger

It performs:

- Tests on Python 3.9, 3.10, 3.11, 3.12
- Code coverage reporting to Codecov
- Linting with ruff
- Format checking with black
- Type checking with mypy

### Documentation Workflow

The `docs.yml` workflow:

- Builds documentation to verify no errors
- Checks for broken links
- Deploys to GitHub Pages on push to main/master

## Release Checklist

Before creating a new release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `HISTORY.rst` or `CHANGELOG.md` with release notes
- [ ] Ensure all tests pass (`pytest tests/`)
- [ ] Ensure code is formatted (`black diwasp tests`)
- [ ] Ensure linting passes (`ruff check diwasp tests`)
- [ ] Build documentation locally (`cd docs && make html`)
- [ ] Update README.md if needed
- [ ] Commit all changes
- [ ] Create and push git tag
- [ ] Create GitHub release
- [ ] Verify PyPI deployment
- [ ] Verify GitHub Pages deployment
- [ ] Test installation: `pip install diwasp`

## Troubleshooting

### PyPI Deployment Fails

1. **Check workflow logs** in GitHub Actions
2. **Verify trusted publishing** is set up correctly on PyPI
3. **Check version number** - can't republish same version
4. **Verify package builds** locally: `python -m build`
5. **Check package with twine**: `twine check dist/*`

### GitHub Pages Build Fails

1. **Check workflow logs** in GitHub Actions
2. **Test locally**: `cd docs && make html`
3. **Check dependencies** in `pyproject.toml` under `[project.optional-dependencies.docs]`
4. **Verify Python version** in workflow matches your local environment
5. **Check GitHub Pages settings** - ensure Source is set to "GitHub Actions"

### Documentation Not Updating

1. **Trigger manual build** via workflow dispatch in Actions tab
2. **Check workflow permissions** in repository settings
3. **Verify the deploy job ran** - it only runs on push to main/master, not on PRs
4. **Check the environment** - ensure `github-pages` environment exists

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for sensitive information
- Prefer trusted publishing over API tokens when possible
- Review all changes in pull requests before merging
- Keep dependencies up to date for security patches
