## Process this file with automake to produce Makefile.in

SUBDIRS = player coach trainer .

noinst_DATA = \
	start.sh.in \
	keepaway.sh.in \
	train.sh.in \
	coach.conf \
	player.conf \
	team_logo.xpm \
	formations-dt/before-kick-off.conf \
	formations-dt/goal-kick-opp.conf \
	formations-dt/goal-kick-our.conf \
	formations-dt/goalie-catch-opp.conf \
	formations-dt/goalie-catch-our.conf \
	formations-dt/goalie-formation.conf \
	formations-dt/indirect-freekick-opp-formation.conf \
	formations-dt/indirect-freekick-our-formation.conf \
	formations-dt/kickin-our-formation.conf \
	formations-dt/normal-formation.conf \
	formations-dt/defense-formation.conf \
	formations-dt/offense-formation.conf \
	formations-dt/setplay-opp-formation.conf \
	formations-dt/setplay-our-formation.conf \
	formations-keeper/before-kick-off.conf \
	formations-keeper/goal-kick-opp.conf \
	formations-keeper/goal-kick-our.conf \
	formations-keeper/goalie-catch-opp.conf \
	formations-keeper/goalie-catch-our.conf \
	formations-keeper/goalie-formation.conf \
	formations-keeper/indirect-freekick-opp-formation.conf \
	formations-keeper/indirect-freekick-our-formation.conf \
	formations-keeper/kickin-our-formation.conf \
	formations-keeper/normal-formation.conf \
	formations-keeper/defense-formation.conf \
	formations-keeper/offense-formation.conf \
	formations-keeper/setplay-opp-formation.conf \
	formations-keeper/setplay-our-formation.conf \
	formations-taker/before-kick-off.conf \
	formations-taker/goal-kick-opp.conf \
	formations-taker/goal-kick-our.conf \
	formations-taker/goalie-catch-opp.conf \
	formations-taker/goalie-catch-our.conf \
	formations-taker/goalie-formation.conf \
	formations-taker/indirect-freekick-opp-formation.conf \
	formations-taker/indirect-freekick-our-formation.conf \
	formations-taker/kickin-our-formation.conf \
	formations-taker/normal-formation.conf \
	formations-taker/defense-formation.conf \
	formations-taker/offense-formation.conf \
	formations-taker/setplay-opp-formation.conf \
	formations-taker/setplay-our-formation.conf

CLEANFILES = *~

DISTCLEANFILES = start.sh train.sh

EXTRA_DIST = $(noinst_DATA) \
	start-debug.sh \
	start-offline.sh

all-local:
	$(LN_S) -f $(top_builddir)/src/player/sample_player $(top_builddir)/src/sample_player
	$(LN_S) -f $(top_builddir)/src/coach/sample_coach $(top_builddir)/src/sample_coach
	$(LN_S) -f $(top_builddir)/src/trainer/sample_trainer $(top_builddir)/src/sample_trainer
