import logging
from flask import Flask, jsonify, Response
from modules.SpotController import SpotController

# Configure explicit, production-grade structured system logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Application Factory Engine. Standardizes object instantiation,
    insulates configurations, and securely binds interface modules.
    """
    application = Flask(__name__, instance_relative_config=True)

    try:
        # Initialize and decouple module layers cleanly
        spot_controller = SpotController()

        # Point straight to our optimized, refactored blueprint singleton property
        application.register_blueprint(spot_controller.blueprint)
        logger.info("Application infrastructure successfully mapped and bound module route: [SpotController]")

    except Exception as initialization_fault:
        logger.critical(f"Fatal application startup disruption during architecture bootstrap: {initialization_fault}")
        raise initialization_fault

    # --- Global Operational Error Gateways ---

    @application.errorhandler(404)
    def page_not_found(error) -> Response:
        """
        Global fallback interceptor for invalid API pathway query calls.
        """
        logger.warning(f"Routing boundary violation: 404 target route not found. Details: {error}")
        return jsonify({
            "success": False,
            "error": "Resource not found",
            "message": "The requested API endpoint pathway does not exist inside our system schema definitions."
        }), 404

    @application.errorhandler(500)
    def internal_server_error(error) -> Response:
        """
        Global safety catchment block protecting live traffic from raw internal unhandled code breaks.
        """
        logger.error(f"Critical internal system exception caught by global boundary: {error}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal pipeline execution failure",
            "message": "An unexpected server-side architectural anomaly occurred. Engineers have been notified."
        }), 500

    return application


# Execute initialization mapping
app = create_app()

if __name__ == "__main__":
    logger.info("Bootstrapping localized multi-tenant portfolio management backend execution environment...")

    # 0.0.0.0 is mandatory to allow proper routing inside Docker/Kubernetes container deployments
    app.run(host="0.0.0.0", port=5000, debug=False)
