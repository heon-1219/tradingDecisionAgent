"""finagent - Financial Product Agent (Financial Product Analyst)."""

import mimetypes

__version__ = "0.1.0"

# The web app (web/app) is served through Starlette's StaticFiles, which asks `mimetypes` for the
# content type. On Windows the registry can map ".js" to text/plain, and browsers refuse to run a
# module script served that way (and, with `X-Content-Type-Options: nosniff`, any script). Pin the
# types the app's files need so the page works the same on every platform.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
