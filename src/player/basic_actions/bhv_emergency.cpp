// -*-c++-*-

/*!
  \file bhv_emergency.cpp
  \brief behaviro for emergency situation.
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 3 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_emergency.h"

#include "bhv_scan_field.h"
#include "view_synch.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

 */
bool
Bhv_Emergency::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  "%s:%d: Bhv_Emergency"
                  ,__FILE__, __LINE__ );

    if ( agent->world().self().viewQuality() != ViewQuality::HIGH )
    {
        agent->setViewAction( new View_Synch() );
    }

    return Bhv_ScanField().execute( agent );
}
