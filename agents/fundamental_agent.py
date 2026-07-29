import yfinance as yf
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("fundamental_agent")

class FundamentalAnalyst:
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.stock = yf.Ticker(ticker_symbol)
        
        # Raw Data Storage
        self.raw_info = {}
        self.raw_financials = pd.DataFrame()
        self.raw_balance_sheet = pd.DataFrame()
        self.raw_fast_info = {}
        
        # Pipeline State
        self.normalized_metrics = {}
        self.computed_metrics = {}
        self.validation_log = []
        self.available_metrics_count = 0
        self.missing_metrics = []

    def fetch_data(self):
        """Step 1: Fetch raw data with independent try-except blocks."""
        print(f"--- [FETCH] Starting data retrieval for {self.ticker_symbol} ---")
        
        # 1. Fetch Info
        try:
            self.raw_info = self.stock.info or {}
            print(f"DEBUG: Info keys count: {len(self.raw_info)}")
        except Exception as e:
            logger.error(f"Error fetching info for {self.ticker_symbol}: {e}")
            self.raw_info = {}

        # 2. Fetch Financials
        try:
            self.raw_financials = self.stock.financials
            if self.raw_financials is None or self.raw_financials.empty:
                self.raw_financials = self.stock.income_stmt
            print(f"DEBUG: Financials shape: {self.raw_financials.shape if self.raw_financials is not None else 'None'}")
        except Exception as e:
            logger.error(f"Error fetching financials for {self.ticker_symbol}: {e}")
            self.raw_financials = pd.DataFrame()

        # 3. Fetch Balance Sheet
        try:
            self.raw_balance_sheet = self.stock.balance_sheet
            print(f"DEBUG: Balance sheet shape: {self.raw_balance_sheet.shape if self.raw_balance_sheet is not None else 'None'}")
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {self.ticker_symbol}: {e}")
            self.raw_balance_sheet = pd.DataFrame()

        # 4. Fetch Fast Info
        try:
            # fast_info is an attribute in newer yfinance versions
            self.raw_fast_info = getattr(self.stock, 'fast_info', {})
            print(f"DEBUG: Fast Info status: {'Available' if self.raw_fast_info else 'Unavailable'}")
        except Exception as e:
            logger.error(f"Error fetching fast_info for {self.ticker_symbol}: {e}")
            self.raw_fast_info = {}

    def _get_val(self, source, keys, default=None):
        """Helper to get value from various dict/DF sources using multiple possible keys."""
        if isinstance(source, pd.DataFrame):
            if source.empty: return default
            idx_map = {str(k).lower().strip().replace(" ", ""): k for k in source.index}
            for k in keys:
                norm_key = k.lower().strip().replace(" ", "")
                if norm_key in idx_map:
                    try:
                        val = source.loc[idx_map[norm_key]]
                        if isinstance(val, pd.Series): return val.iloc[0]
                        return val
                    except: continue
        elif isinstance(source, (dict, object)):
            # Handle both dict and yfinance.FastInfo object
            for k in keys:
                try:
                    val = source.get(k) if isinstance(source, dict) else getattr(source, k, None)
                    if val is not None: return val
                except: continue
        return default

    def normalize_data(self):
        """Step 2: Normalize raw data into standardized keys."""
        print(f"--- [NORMALIZE] Transforming data for {self.ticker_symbol} ---")
        
        # Define Mappings
        mappings = {
            "revenue": ["totalRevenue", "Total Revenue", "Operating Revenue", "Interest Income", "Total Interest Income", "Net Interest Income"],
            "equity": ["totalStockholderEquity", "Total Stockholders Equity", "Stockholders Equity", "Common Stock Equity", "Total Equity", "Total Equity Gross Minority Interest"],
            "net_income": ["netIncomeToCommon", "Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"],
            "roe": ["returnOnEquity", "return_on_equity"],
            "debt": ["totalDebt", "Total Debt", "Net Debt", "Long Term Debt"],
            "op_margin": ["operatingMargins", "Operating Margin"],
            "rev_growth": ["revenueGrowth", "Revenue Growth"]
        }

        # Priority: Financials/Balance Sheet first, then Info fallback
        self.normalized_metrics = {
            "revenue_data": self._get_val(self.raw_financials, mappings["revenue"]), # Full series for growth
            "equity": self._get_val(self.raw_balance_sheet, mappings["equity"]),
            "net_income": self._get_val(self.raw_financials, mappings["net_income"]),
            "debt": self._get_val(self.raw_balance_sheet, mappings["debt"]),
            "roe_raw": self._get_val(self.raw_info, mappings["roe"]),
            "op_margin_raw": self._get_val(self.raw_info, mappings["op_margin"]),
            "rev_growth_raw": self._get_val(self.raw_info, mappings["rev_growth"])
        }

        # Backup from Info/FastInfo if still missing
        if self.normalized_metrics["debt"] is None:
            self.normalized_metrics["debt"] = self._get_val(self.raw_info, ["totalDebt", "debt"])
        
        print(f"DEBUG: Normalized keys: {list(self.normalized_metrics.keys())}")

    def validate_and_compute(self):
        """Step 3 & 4: Validate inputs and compute derived metrics."""
        print(f"--- [COMPUTE] Validating and deriving metrics for {self.ticker_symbol} ---")
        
        # Sector Detection (Banking Specific)
        sector = self.raw_info.get("sector", "Unknown")
        industry = self.raw_info.get("industry", "Unknown")
        self.is_bank = (sector == "Financial Services" or "Bank" in industry)
        print(f"DEBUG: Sector: {sector}, Industry: {industry}, IsBank: {self.is_bank}")

        m = self.normalized_metrics
        c = {}

        # 1. ROE (Compute if raw missing)
        roe = m["roe_raw"]
        if roe is None and m["net_income"] is not None and m["equity"] is not None and m["equity"] != 0:
            roe = m["net_income"] / m["equity"]
        c["roe"] = roe if roe is not None and not np.isnan(roe) else None

        # 2. Debt to Equity (Compute if raw missing)
        de = self._get_val(self.raw_info, ["debtToEquity"])
        if de is not None:
            de = de / 100.0 if de > 5 else de # Normalize to decimal
        elif m["debt"] is not None and m["equity"] is not None and m["equity"] != 0:
            de = m["debt"] / m["equity"]
        c["debt_to_equity"] = de if de is not None and not np.isnan(de) else None

        # 3. Revenue Growth
        rev_growth = m["rev_growth_raw"]
        rev_series = m["revenue_data"]
        # If we have a series from financials, calculate manual growth
        if rev_growth is None and isinstance(rev_series, pd.Series) and len(rev_series) >= 2:
            current, prev = rev_series.iloc[0], rev_series.iloc[1]
            if prev and prev != 0:
                rev_growth = (current - prev) / prev
        c["revenue_growth"] = rev_growth if rev_growth is not None and not np.isnan(rev_growth) else None

        # 4. Operating Margin
        margin = m["op_margin_raw"]
        if margin is None and not self.is_bank: # Don't compute op margin for banks, use interest income logic
             rev = self._get_val(self.raw_financials, ["Total Revenue", "Operating Revenue"])
             op_inc = self._get_val(self.raw_financials, ["Operating Income"])
             if op_inc and rev and rev != 0:
                 margin = op_inc / rev
        c["operating_margin"] = margin if margin is not None and not np.isnan(margin) else None

        # 5. PE Ratio
        c["pe_ratio"] = self.raw_info.get("trailingPE") or self.raw_info.get("forwardPE")

        self.computed_metrics = c
        self.missing_metrics = [k for k, v in c.items() if v is None]
        self.available_metrics_count = len(c) - len(self.missing_metrics)
        print(f"DEBUG: Computed metrics: {self.computed_metrics}")

    def score_and_aggregate(self):
        """Step 5 & 6: Partial scoring and final aggregation."""
        print(f"--- [SCORE] Aggregating final results for {self.ticker_symbol} ---")
        
        c = self.computed_metrics
        score_weight = 0
        total_possible_weight = 0

        # ROE Scoring
        if c["roe"] is not None:
            total_possible_weight += 1
            if c["roe"] > 0.15: score_weight += 1
            elif c["roe"] > 0.08: score_weight += 0.5
        
        # Debt Scoring (Bank Adjusted)
        if c["debt_to_equity"] is not None:
            total_possible_weight += 1
            threshold = 2.5 if self.is_bank else 1.0
            if c["debt_to_equity"] < threshold: score_weight += 1
            elif c["debt_to_equity"] < (threshold * 1.5): score_weight += 0.5

        # Revenue Growth Scoring
        if c["revenue_growth"] is not None:
            total_possible_weight += 1
            if c["revenue_growth"] > 0.12: score_weight += 1
            elif c["revenue_growth"] > 0.05: score_weight += 0.6
            elif c["revenue_growth"] > 0: score_weight += 0.3

        # Margin Scoring
        if c["operating_margin"] is not None:
            total_possible_weight += 1
            if c["operating_margin"] > 0.15: score_weight += 1
            elif c["operating_margin"] > 0.08: score_weight += 0.5

        # Final Score Calculation
        if total_possible_weight == 0:
            final_score = 3.0 # Neutral safety baseline as per requirement
        else:
            final_score = (score_weight / total_possible_weight) * 10
        
        # Enforcement of Rule 7: Never return 0.0 unless explicit negative
        if final_score < 3.0:
            # Check if fundamentals are actually negative
            is_bad = (c["roe"] is not None and c["roe"] < 0) or \
                     (c["revenue_growth"] is not None and c["revenue_growth"] < -0.1)
            if not is_bad:
                final_score = 3.0

        # Confidence Level Calculation
        if self.available_metrics_count >= 4: confidence = "high"
        elif self.available_metrics_count >= 2: confidence = "medium"
        else: confidence = "low"

        # Format Validation for UI
        formatted_metrics = {}
        for k, v in self.computed_metrics.items():
            if v is None:
                formatted_metrics[k] = "Unavailable"
            elif k in ["roe", "revenue_growth", "operating_margin"]:
                formatted_metrics[k] = f"{round(v * 100, 2):.2f}%"
            else:
                formatted_metrics[k] = str(round(v, 2))

        return {
            "agent": "fundamental",
            "ticker": self.ticker_symbol,
            "score": round(final_score, 1),
            "fundamental_score": round(final_score, 1), # Compatibility
            "metrics": {
                "long_name": self.raw_info.get("longName") or self.ticker_symbol,
                **formatted_metrics
            },
            "available_metrics": self.available_metrics_count,
            "missing_metrics": self.missing_metrics,
            "data_confidence": confidence
        }

def analyze_fundamentals(ticker: str) -> dict:
    """Entry point for the fundamental agent pipeline."""
    analyst = FundamentalAnalyst(ticker)
    try:
        analyst.fetch_data()
        analyst.normalize_data()
        analyst.validate_and_compute()
        return analyst.score_and_aggregate()
    except Exception as e:
        logger.error(f"Critical failure in fundamental pipeline for {ticker}: {e}")
        # Final safety net returning a baseline response instead of crashing
        return {
            "agent": "fundamental",
            "ticker": ticker,
            "score": 3.0,
            "fundamental_score": 3.0,
            "metrics": {"long_name": ticker},
            "available_metrics": 0,
            "missing_metrics": ["Pipeline Error"],
            "data_confidence": "none"
        }


