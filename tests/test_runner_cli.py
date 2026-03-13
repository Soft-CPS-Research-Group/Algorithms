from run_experiment import build_argument_parser


def test_cli_accepts_job_id_dash_variant():
    parser = build_argument_parser()
    args = parser.parse_args(["--config", "cfg.yaml", "--job-id", "job-123"])
    assert args.job_id == "job-123"


def test_cli_accepts_job_id_underscore_variant():
    parser = build_argument_parser()
    args = parser.parse_args(["--config", "cfg.yaml", "--job_id", "job-456"])
    assert args.job_id == "job-456"
