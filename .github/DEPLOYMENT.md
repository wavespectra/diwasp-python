# Deployment Guide

This document describes how to deploy diwasp-python to PyPI and Read the Docs.

## PyPI Deployment

### Prerequisites

1. **Set up PyPI Trusted Publishing** (recommended):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher for your GitHub repository
   - Repository: `yourusername/diwasp-python`
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
   - Go to https://github.com/yourusername/diwasp-python/releases/new
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

## Read the Docs Deployment

### Initial Setup

1. **Import the project**:
   - Go to https://readthedocs.org/dashboard/import/
   - Click "Import a Repository"
   - Select `diwasp-python` from your GitHub repositories
   - Click "Next"

2. **Configure the project**:
   - The `.readthedocs.yaml` file will automatically configure the build
   - No additional configuration needed in the Read the Docs dashboard

3. **Verify webhook**:
   - Go to your GitHub repository Settings → Webhooks
   - You should see a webhook for `https://readthedocs.org/api/v2/webhook/...`
   - If not present, Read the Docs will add it automatically on first build

### Building Documentation

Documentation is built automatically:

- **On every push to main/master**: Full documentation rebuild
- **On pull requests**: Preview builds for review
- **Manual trigger**: Use the "Build Version" button in Read the Docs dashboard

### Accessing Documentation

Once deployed, documentation will be available at:

- Latest: https://diwasp-python.readthedocs.io/en/latest/
- Stable: https://diwasp-python.readthedocs.io/en/stable/
- Specific version: https://diwasp-python.readthedocs.io/en/v0.2.0/

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
- Uploads artifacts for review
- Provides information about Read the Docs integration

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
- [ ] Verify Read the Docs build
- [ ] Test installation: `pip install diwasp`

## Troubleshooting

### PyPI Deployment Fails

1. **Check workflow logs** in GitHub Actions
2. **Verify trusted publishing** is set up correctly on PyPI
3. **Check version number** - can't republish same version
4. **Verify package builds** locally: `python -m build`
5. **Check package with twine**: `twine check dist/*`

### Read the Docs Build Fails

1. **Check build logs** on Read the Docs dashboard
2. **Verify `.readthedocs.yaml`** configuration
3. **Test locally**: `cd docs && make html`
4. **Check dependencies** in `pyproject.toml` under `[project.optional-dependencies.docs]`
5. **Verify Python version** matches `.readthedocs.yaml`

### Documentation Not Updating

1. **Trigger manual build** on Read the Docs
2. **Check webhook** is active in GitHub settings
3. **Verify branch** is set to build in Read the Docs settings
4. **Clear cache** in Read the Docs admin panel

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for sensitive information
- Prefer trusted publishing over API tokens when possible
- Review all changes in pull requests before merging
- Keep dependencies up to date for security patches
