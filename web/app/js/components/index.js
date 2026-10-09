/* The conversation components (Task 4.2). Importing a module registers it with the shell:
 * stream-event listeners and panel views in registry.js, markdown post-processors in markdown.js,
 * and bus listeners in state.js. The order of these imports does not matter.
 *
 *   toolcard    tool-input-available / tool-output-available   tool cards (replaces the tool rows)
 *   thinking    reasoning-delta                                 the Thinking block
 *   sources     [E#] chips, the Sources row, the source popover, GET /api/sources/{session}
 *   grounding   source-grounding (sandboxed iframe), data-grounding-flags (advisory note)
 *   mandate     data-mandate (draft card), the Mandate panel
 *   ticket      data-trade-proposal and a reloaded chat's tickets: the ticket card, its timeline
 *   portfolio   the Portfolio panel
 *   orders      the Orders panel
 *   settings    the Settings panel (keys, trading switch, kill switch)
 */
import './toolcard.js';
import './thinking.js';
import './sources.js';
import './grounding.js';
import './mandate.js';
import './ticket-thread.js';
import './portfolio.js';
import './orders.js';
import './settings.js';
