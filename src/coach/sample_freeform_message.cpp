// -*-c++-*-

/*!
  \file sample_freeform_message.cpp
  \brief freeform message builder Source File
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

#include "sample_freeform_message.h"

#include <rcsc/common/player_param.h>
#include <rcsc/types.h>

#include <cstring>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

 */
OpponentPlayerTypeMessage::OpponentPlayerTypeMessage( const int t1,
                                                      const int t2,
                                                      const int t3,
                                                      const int t4,
                                                      const int t5,
                                                      const int t6,
                                                      const int t7,
                                                      const int t8,
                                                      const int t9,
                                                      const int t10,
                                                      const int t11 )
    : FreeformMessage( "pt" )
{
    int i = 0;
    M_player_type_id[i++] = t1;
    M_player_type_id[i++] = t2;
    M_player_type_id[i++] = t3;
    M_player_type_id[i++] = t4;
    M_player_type_id[i++] = t5;
    M_player_type_id[i++] = t6;
    M_player_type_id[i++] = t7;
    M_player_type_id[i++] = t8;
    M_player_type_id[i++] = t9;
    M_player_type_id[i++] = t10;
    M_player_type_id[i++] = t11;

    buildMessage();
}

/*-------------------------------------------------------------------*/
/*!

 */
void
OpponentPlayerTypeMessage::setPlayerType( const int unum,
                                          const int id )
{
    if ( unum < 1 || 11 < unum )
    {
        std::cerr << __FILE__ << ' ' << __LINE__
                  << ": illegal uniform number " << unum
                  << std::endl;
        return;
    }

    M_player_type_id[unum-1] = id;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
OpponentPlayerTypeMessage::buildMessage()
{
    M_message.clear();

    const int player_types = PlayerParam::i().playerTypes();
    for ( int i = 0; i < 11; ++i )
    {
        int id = M_player_type_id[i];

        if ( id < Hetero_Unknown || player_types <= id )
        {
            std::cerr << __FILE__ << ' ' << __LINE__
                      << ": illegal player type id " << id
                      << std::endl;
            id = Hetero_Unknown;
        }

        if ( id == Hetero_Unknown )
        {
            M_message += '-';
        }
        else
        {
            M_message += static_cast< char >( 'A' + id );
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
OpponentPlayerTypeMessage::append( std::string & to ) const
{
    if ( M_message.empty() )
    {
        return false;
    }

    to += '(';
    to += this->type();
    to += ' ';
    to += M_message;
    to += ')';

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
std::ostream &
OpponentPlayerTypeMessage::printDebug( std::ostream & os ) const
{
    os << "[PlayerType]";
    return os;
}
