import logging
from flask import Blueprint, request, jsonify, Response
from services.data_analysis.SpotService import SpotServices
from services.data_transformation.CandlestickParser import CandlestickParser

logger = logging.getLogger(__name__)


class SpotController:
    """
    Production-grade controller managing HTTP routing gates for historical spot execution markets.
    Utilizes decoupled dependency injection to insulate business engine workflows.
    """

    def __init__(self, spot_service: SpotServices = None, parser: CandlestickParser = None):
        """
        Initializes the Controller layer. Reuses singletons to optimize memory footprints under scale.
        """
        self.spot_service = spot_service or SpotServices()
        self.parser = parser or CandlestickParser()
        self.blueprint = Blueprint(name='spot', import_name=__name__, url_prefix='/spot')

        # Register execution routes automatically upon initialization
        self._register_routes()

    def _register_routes(self) -> None:
        """
        Binds functional execution pathways tightly inside the Blueprint routing pool.
        """

        @self.blueprint.route('/test', methods=['GET'])
        def timeline() -> Response:
            return jsonify({"msg": 'hello from spot'})

        @self.blueprint.route('/last_historical_spot', methods=['GET'])
        def last_historical_spot() -> Response:
            market = request.args.get('market', '').strip()
            currency = request.args.get('currency', '').strip()
            interval = request.args.get('interval', '').strip()

            # Strict guard protection layout to prevent parameter mapping exceptions
            if not market or not currency or not interval:
                return jsonify({"error": "Missing mandatory parameters: [market, currency, interval]"}), 400

            try:
                # Execute decoupled data retrieval using our initialized singletons
                spot_data = self.spot_service.get_spot_api(
                    market=market.upper(),
                    currency=currency,
                    interval=interval
                )

                parsed_data = self.parser.parse_spot_by_market_currency(spot_data, market, currency)
                return jsonify({'data': parsed_data})

            except Exception as execution_error:
                logger.error(f"Endpoint Exception in last_historical_spot: {execution_error}", exc_info=True)
                return jsonify({"error": "Internal pipeline execution failure during spot retrieval"}), 500

        @self.blueprint.route('/historical_spot', methods=['GET'])
        def historical_spot() -> Response:
            market = request.args.get('market', '').strip()
            currency = request.args.get('currency', '').strip()
            interval = request.args.get('interval', '').strip()
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')

            if not market or not currency or not interval:
                return jsonify({"error": "Missing mandatory validation parameters"}), 400

            try:
                spot_data_all = self.spot_service.get_spot_api(
                    market=market.upper(),
                    currency=currency,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )

                parsed_binance_data = []
                for spot_data in spot_data_all:
                    transformed_chunk = self.parser.parse_spot_by_market_currency(spot_data, market, currency)
                    # .extend() mutates the array in-place, drastically accelerating performance execution
                    parsed_binance_data.extend(transformed_chunk)

                return jsonify({'data': parsed_binance_data})

            except Exception as execution_error:
                logger.error(f"Endpoint Exception inside historical_spot sequence: {execution_error}", exc_info=True)
                return jsonify({"error": "Internal pipeline execution failure during batch spot retrieval"}), 500
