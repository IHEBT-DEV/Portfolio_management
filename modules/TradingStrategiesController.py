import logging
from flask import Blueprint, request, jsonify, Response
# Import your internal services safely using clean lower-case snake_case imports
from services.data_analysis.SpotService import SpotServices
from services.data_transformation.CandlestickParser import CandlestickParser

logger = logging.getLogger(__name__)


class TradingStrategiesController:
    """
    Production-grade controller orchestrating automated quantitative trading strategy pathways.
    Processes historical candlestick series data to formulate trend strategy evaluation signals.
    """

    def __init__(self, spot_service: SpotServices = None, parser: CandlestickParser = None):
        """
        Initializes the Trading Strategies Gateway. Reuses infrastructure singletons.
        """
        self.spot_service = spot_service or SpotServices()
        self.parser = parser or CandlestickParser()
        self.blueprint = Blueprint(name='strategies', import_name=__name__, url_prefix='/strategies')

        # Register functional execution pathways automatically upon startup
        self._register_routes()

    def _register_routes(self) -> None:
        """
        Binds API routing handlers inside the core Blueprint configuration matrix.
        """

        @self.blueprint.route('/moving_average_cross', methods=['GET'])
        def moving_average_cross() -> Response:
            """
            Evaluates a Simple Moving Average crossover strategy (Fast SMA vs Slow SMA).
            Returns an actionable execution signal mapping (BUY, SELL, HOLD) for the asset.
            """
            market = request.args.get('market', 'BINANCE').strip().upper()
            currency = request.args.get('currency', '').strip().upper()
            interval = request.args.get('interval', '1D').strip()

            # Fetch windows with strict integer validation default assignments
            try:
                fast_period = int(request.args.get('fast_period', 5))
                slow_period = int(request.args.get('slow_period', 20))
            except ValueError:
                return jsonify({"error": "Strategy execution aborted: Allocation windows must be valid integers."}), 400

            if not currency:
                return jsonify({"error": "Missing mandatory trading asset parameter: [currency]"}), 400

            try:
                # 1. Fetch raw matrix candlestick inputs from the data core
                raw_spot_data = self.spot_service.get_spot_api(
                    market=market,
                    currency=currency,
                    interval=interval
                )

                # 2. Map and parse inputs cleanly into our standard unified schema models
                parsed_candles = self.parser.parse_spot_by_market_currency(raw_spot_data, market, currency)

                # Validation safeguard: Ensure the series data contains enough rows to evaluate moving averages
                if len(parsed_candles) < slow_period:
                    return jsonify({
                        "success": False,
                        "signal": "HOLD",
                        "reason": f"Insufficient dataset series size. Core requires at least {slow_period} candles, matched only {len(parsed_candles)}."
                    })

                # 3. Quantitative Math Execution: Extract closing prices (newest candle is at the end of the array)
                closing_prices = [candle['close'] for candle in parsed_candles]

                # Compute localized moving average intersections
                fast_sma = sum(closing_prices[-fast_period:]) / fast_period
                slow_sma = sum(closing_prices[-slow_period:]) / slow_period

                # 4. Enforce Trend Strategy Evaluation Logic
                if fast_sma > slow_sma:
                    signal = "BUY"
                    reason = f"Bullish Crossover: Fast SMA ({fast_sma:.4f}) crossed above Slow SMA ({slow_sma:.4f})."
                elif fast_sma < slow_sma:
                    signal = "SELL"
                    reason = f"Bearish Breakdown: Fast SMA ({fast_sma:.4f}) dropped below Slow SMA ({slow_sma:.4f})."
                else:
                    signal = "HOLD"
                    reason = "Static equilibrium: Indicators tracking completely neutral variance metrics."

                return jsonify({
                    "success": True,
                    "market": market,
                    "currency": currency,
                    "interval": interval,
                    "metrics": {
                        "fast_sma": fast_sma,
                        "slow_sma": slow_sma,
                        "total_candles_evaluated": len(parsed_candles)
                    },
                    "signal": signal,
                    "reason": reason
                })

            except Exception as strategy_anomaly:
                logger.error(f"Strategy evaluation exception inside moving_average_cross route: {strategy_anomaly}",
                             exc_info=True)
                return jsonify({"error": "Internal analytical engine runtime break encountered."}), 500
