// -*-c++-*-

/*!
  \file freeform_parser.cpp
  \brief freeform message parser Source File
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

#include "sample_freeform_message_parser.h"

#include <rcsc/common/player_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/player/world_model.h>

#include <cstring>

//#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#include <sstream>
#endif

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
OpponentPlayerTypeMessageParser::OpponentPlayerTypeMessageParser( WorldModel & world )
    : FreeformMessageParser( "pt" ),
      M_world( world )
{

}

/*-------------------------------------------------------------------*/
/*!

*/
int
OpponentPlayerTypeMessageParser::parse( const char * msg )
{
    char buf[16];
    int n_read = 0;

    std::memset( buf, 0, 16 );

    if ( std::sscanf( msg, " ( pt %15[^)] ) %n", buf, &n_read ) != 1 )
    {
        std::cerr << __FILE__ << ' ' << __LINE__
                  << ": could not read the message. [" << msg << "]"
                  << std::endl;
        return -1;
    }

    if ( std::strlen( buf ) != 11 )
    {
        std::cerr << __FILE__ << ' ' << __LINE__
                  << ": (OpponentPlayerTypeMesageParser) illegal message [" << buf << "]"
                  << std::endl;
        return -1;
    }

    const int player_types = PlayerParam::i().playerTypes();

#ifdef DEBUG_PRINT
    std::ostringstream ostr;
    std::cerr << "parsed player types: ";
#endif

    for ( int i = 0; i < 11; ++i )
    {
        if ( buf[i] == '-' )
        {
            M_world.setTheirPlayerType( i + 1, Hetero_Unknown );
#ifdef DEBUG_PRINT
            ostr << '(' << i + 1 << ' ' << Hetero_Unknown << ')';
            std::cerr << '(' << i + 1 << ' ' << Hetero_Unknown << ')';
#endif
        }
        else
        {
            int id = static_cast< int >( buf[i] ) - static_cast< int >( 'A' );
            if ( id < 0 || player_types <= id )
            {
                std::cerr << __FILE__ << ' ' << __LINE__
                          << ": illegal player type id."
                          << " char=" << buf[i]
                          << " id=" << id
                          << std::endl;
                return -1;
            }

            M_world.setTheirPlayerType( i + 1, id );
#ifdef DEBUG_PRINT
            ostr << '(' << i + 1 << ' ' << id << ')';
            std::cerr << '(' << i + 1 << ' ' << id << ')';
#endif
        }
    }

#ifdef DEBUG_PRINT
    dlog.addText( Logger::SENSOR,
                  __FILE__": opponent player type %s", ostr.str().c_str() );
    std::cerr << std::endl;
#endif

    return n_read;
}
