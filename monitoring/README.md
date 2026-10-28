# Cost monitoring dashboard

`cost_monitoring_dashboard.html` is a **static viewer**. It never talks to AWS and has no
credential fields. It only renders a `cost_report.json` file that you generate on your own
machine with `scripts/cost_monitor.py`.

## 1. Generate the report

Run this from the repository root, using the same AWS credentials you use for the AWS CLI:

```bash
# SSO / IAM Identity Center
aws sso login --profile research
AWS_PROFILE=research python scripts/cost_monitor.py --json monitoring/cost_report.json

# Options
python scripts/cost_monitor.py --json monitoring/cost_report.json --days 30 --region us-west-2
```

The script uses the standard boto3 credential chain (`AWS_PROFILE`, SSO, environment
variables, or an instance role) and never asks for access keys. It needs read-only
permissions:

- `ce:GetCostAndUsage`: month-to-date spend, the daily trend, and cost by instance type.
  Cost Explorer data covers the whole account and charges about $0.01 per request
  (two requests per report).
- `ec2:DescribeInstances`: running ML instances in `--region`, meaning instances tagged
  `Name=ML-*` or `AutoTerminate=true` (the same rule as `scripts/emergency_shutdown.py`).

The report contains:

| Key | Contents |
|---|---|
| `generated_at`, `region`, `currency` | when, where, and the unit (USD) |
| `month_to_date` | `total`, `trainium`, `inferentia`, `other`, `by_instance_type`, and the date range (end is exclusive, so today's partial spend is left out) |
| `daily_costs` | `[{date, amount}]` for the last `--days` completed days (default 14), all services |
| `running_instances` | id, name, type, family, lifecycle (spot/on-demand), launch time, uptime |
| `summary` | instance counts, 7-day average daily spend, projected month total |

The burn rate and projection come from **billed** spend, not a hard-coded price table.
Cost Explorer data can lag by up to about 24 hours.

`*.json` is already in `.gitignore`, so the report (which holds your account's cost data)
won't be committed by accident.

## 2. Open the dashboard

Pick one:

- **Open the file directly.** Open `monitoring/cost_monitoring_dashboard.html` in a browser,
  then click the file picker or drag `cost_report.json` onto the page.
- **Serve the folder locally.** The page then loads `cost_report.json` by itself:

  ```bash
  python -m http.server -d monitoring 8000
  # open http://localhost:8000/cost_monitoring_dashboard.html
  ```

To refresh, run the `--json` command again and click **Reload report**. There is no
auto-refresh. The page shows a warning when the loaded report is more than 24 hours old.

The trend chart uses Chart.js from a CDN. When you're offline, the same numbers are still
in the "Show as table" view.

To stop instances, use `python scripts/emergency_shutdown.py`. The dashboard is read-only
on purpose.

## Why credentials never go in the browser

An earlier version of this dashboard had you paste an access key ID and secret key into a
web form, then called Cost Explorer, EC2, and S3 from the page with the AWS JavaScript SDK.
That is a bad pattern:

- **Long-lived keys end up in the browser.** Browser extensions, saved form data, devtools,
  a shared screen, or an XSS bug on the page can all expose them, and an IAM user key stays
  valid until someone notices and rotates it.
- **The permissions are too broad.** A page that can list *and terminate* instances needs
  write access to EC2, far more than viewing costs requires.
- **It teaches the wrong habit.** The recommended way to authenticate is short-lived
  credentials from IAM Identity Center (SSO) or an instance role, resolved by the CLI or
  SDK on your machine. Access keys typed into an application should not be part of that.

Now the credentials stay with the AWS CLI and boto3. The script exports a plain data file,
and the browser only reads that file.
