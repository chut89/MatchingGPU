/*
  Flycpature camera
  Copyright (C) 2011  Martin Vogt

  This program is licensed under the DFKI probitary license.
  All rights reserved.

  For more information look at the file COPYRIGHT in this package
 */


#ifndef __CAM_H
#define __CAM_H

#include <FlyCapture2.h>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <CameraBase.h>

class call_back_data
{
public:
    // camera number
    int num;
    int max_number;
    std::string number_as_string;
};


class cam
{

public:
    enum pgr_registers
    {
        ePGR_V_FORMAT_INQ = 0x100,
        ePGR_V_MODE_INQ_7 = 0x19c,
        ePGR_FEATURE_HI_INQ = 0x404,
        ePGR_IMAGE_DATA_FORMAT = 0x1048,
        ePGR_BAYER_TILE_MAPPING = 0x1040
    };
    enum reg_index
    {
        ePGRI_BRIGHTNESS = 0,
        ePGRI_AUTO_EXPOSURE,
        ePGRI_SHARPNESS,
        ePGRI_WHITE_BALANCE,
        ePGRI_HUE,
        ePGRI_SATURATION,
        ePGRI_GAMMA,
        ePGRI_SHUTTER,
        ePGRI_GAIN,
        ePGRI_IRIS,
        ePGRI_FOCUS,
        ePGRI_TEMPERATURE,
        ePGRI_TRIGGER,
        ePGRI_TRIGGER_DELAY,
        ePGRI_WHITE_SHADING,
        ePGRI_FRAME_RATE,
        ePGRI_BAYER_MONO_CTRL = 24,
        ePGRI_Y16_DATA_FORMAT = 31
    };
    enum bayer_pattern
    {
        ePGRB_RED = 0x52,
        ePGRB_GREEN = 0x47,
        ePGRB_BLUE = 0x42,
        ePGRB_MONO = 0x59
    };
    enum grab_modes
    {
        eGRAB_MODE_OFF,
        eGRAB_MODE_ON,
        eGRAB_MODE_REQUEST,
        eGRAB_MODE_DIMENSION
    };


    cam ();
    virtual ~cam();
    void start_capture ( FlyCapture2::ImageEventCallback fn );
    bool start_capture ( std::vector<FlyCapture2::Image*> render_images );
    void stop_capture();
    void grab ( std::vector<FlyCapture2::Image*> render_images );
    int get_number_of_cameras()
    {
        return number_of_cameras;
    }

protected:


private:

    void setDefaultProperties();

    void print_camera_info ( FlyCapture2::CameraInfo* pCamInfo );
    void print_format7_image_settings ( FlyCapture2::Format7ImageSettings * s );
    void print_format7_info ( FlyCapture2::Format7Info * s );
    void print_format7_packet_info ( FlyCapture2::Format7PacketInfo * s );
    void print_config ( FlyCapture2::FC2Config * s );
    void print_trigger_mode ( FlyCapture2::TriggerMode * s );
    void print_embedded_image_info ( FlyCapture2::EmbeddedImageInfo* s );
    void print_property ( FlyCapture2::Property* s );
    void print_property_info ( FlyCapture2::PropertyInfo* s );

    inline void handle_error ( FlyCapture2::Error error )
    {
        if ( error != FlyCapture2::PGRERROR_OK )
        {
            const char* msg;
            printf ( "%i %s\n", ( int ) error.GetType(), error_type_to_char ( error.GetType() ) );
            msg=error.CollectSupportInformation();
            if ( msg == NULL )
            {
                msg="NULL PTR";
            }
            printf ( "%s\n", msg );

            //printf("%s >> PrintErrorTrace\n", Description());
            //error.PrintErrorTrace();
            //printf("%s >> GetDescription\n", Description());
            //printf("%s\n", error.GetDescription());
            //printf("%s >> done.\n", Description());

            exit ( 1 );
        }
    };

    int bayer_mapping ( unsigned int b )
    {
        switch ( b )
        {
        case 0x47524247: // GRBG
            return CV_BayerGB2BGR;
        case 0x52474742: // RGGB
            return CV_BayerBG2BGR;
        default:
            printf ( " >> unknown pattern %i\n", b );
        };
        return CV_BayerGB2BGR;
    }

    const char * error_type_to_char ( FlyCapture2::ErrorType e )
    {
        switch ( e )
        {
        case  FlyCapture2::PGRERROR_UNDEFINED :
            return "< Undefined ";
        case  FlyCapture2::PGRERROR_OK:
            return "< Function returned with no errors. ";
        case  FlyCapture2::PGRERROR_FAILED:
            return "< General failure. ";
        case  FlyCapture2::PGRERROR_NOT_IMPLEMENTED:
            return "< Function has not been implemented. ";
        case  FlyCapture2::PGRERROR_FAILED_BUS_MASTER_CONNECTION:
            return "< Could not connect to Bus Master. ";
        case  FlyCapture2::PGRERROR_NOT_CONNECTED:
            return "< Camera has not been connected. ";
        case  FlyCapture2::PGRERROR_INIT_FAILED:
            return "< Initialization failed. ";
        case  FlyCapture2::PGRERROR_NOT_INTITIALIZED:
            return "< Camera has not been initialized. ";
        case  FlyCapture2::PGRERROR_INVALID_PARAMETER:
            return "< Invalid parameter passed to function. ";
        case  FlyCapture2::PGRERROR_INVALID_SETTINGS:
            return "< Setting set to camera is invalid. ";
        case  FlyCapture2::PGRERROR_INVALID_BUS_MANAGER:
            return "< Invalid Bus Manager object. ";
        case  FlyCapture2::PGRERROR_MEMORY_ALLOCATION_FAILED:
            return "< Could not allocate memory. ";
        case  FlyCapture2::PGRERROR_LOW_LEVEL_FAILURE:
            return "< Low level error. ";
        case  FlyCapture2::PGRERROR_NOT_FOUND:
            return "< Device not found. ";
        case  FlyCapture2::PGRERROR_FAILED_GUID:
            return "< GUID failure. ";
        case  FlyCapture2::PGRERROR_INVALID_PACKET_SIZE:
            return "< Packet size set to camera is invalid. ";
        case  FlyCapture2::PGRERROR_INVALID_MODE:
            return "< Invalid mode has been passed to function. ";
        case  FlyCapture2::PGRERROR_NOT_IN_FORMAT7:
            return "< Error due to not being in Format7. ";
        case  FlyCapture2::PGRERROR_NOT_SUPPORTED:
            return "< This feature is unsupported. ";
        case  FlyCapture2::PGRERROR_TIMEOUT:
            return "< Timeout error. ";
        case  FlyCapture2::PGRERROR_BUS_MASTER_FAILED:
            return "< Bus Master Failure. ";
        case  FlyCapture2::PGRERROR_INVALID_GENERATION:
            return "< Generation Count Mismatch. ";
        case  FlyCapture2::PGRERROR_LUT_FAILED:
            return "< Look Up Table failure. ";
        case  FlyCapture2::PGRERROR_IIDC_FAILED:
            return "< IIDC failure. ";
        case  FlyCapture2::PGRERROR_STROBE_FAILED:
            return "< Strobe failure. ";
        case  FlyCapture2::PGRERROR_TRIGGER_FAILED:
            return "< Trigger failure. ";
        case  FlyCapture2::PGRERROR_PROPERTY_FAILED:
            return "< Property failure. ";
        case  FlyCapture2::PGRERROR_PROPERTY_NOT_PRESENT:
            return "< Property is not present. ";
        case  FlyCapture2::PGRERROR_REGISTER_FAILED:
            return "< Register access failed. ";
        case  FlyCapture2::PGRERROR_READ_REGISTER_FAILED:
            return "< Register read failed. ";
        case  FlyCapture2::PGRERROR_WRITE_REGISTER_FAILED:
            return "< Register write failed. ";
        case  FlyCapture2::PGRERROR_ISOCH_FAILED:
            return "< Isochronous failure. ";
        case  FlyCapture2::PGRERROR_ISOCH_ALREADY_STARTED:
            return "< Isochronous transfer has already been started. ";
        case  FlyCapture2::PGRERROR_ISOCH_NOT_STARTED:
            return "< Isochronous transfer has not been started. ";
        case  FlyCapture2::PGRERROR_ISOCH_START_FAILED:
            return "< Isochronous start failed. ";
        case  FlyCapture2::PGRERROR_ISOCH_RETRIEVE_BUFFER_FAILED:
            return "< Isochronous retrieve buffer failed. ";
        case  FlyCapture2::PGRERROR_ISOCH_STOP_FAILED:
            return "< Isochronous stop failed. ";
        case  FlyCapture2::PGRERROR_ISOCH_SYNC_FAILED:
            return "< Isochronous image synchronization failed. ";
        case  FlyCapture2::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED:
            return "< Isochronous bandwidth exceeded. ";
        case  FlyCapture2::PGRERROR_IMAGE_CONVERSION_FAILED:
            return "< Image conversion failed. ";
        case  FlyCapture2::PGRERROR_IMAGE_LIBRARY_FAILURE:
            return "< Image library failure. ";
        case  FlyCapture2::PGRERROR_BUFFER_TOO_SMALL:
            return "< Buffer is too small. ";
        case  FlyCapture2::PGRERROR_IMAGE_CONSISTENCY_ERROR:
            return "< There is an image consistency error. ";
        default:
            return "not known";
        }
    }


    const char * property_type_to_char ( FlyCapture2::PropertyType p )
    {
        switch ( p )
        {
        case FlyCapture2::BRIGHTNESS:
            return "Brightness";
        case FlyCapture2::AUTO_EXPOSURE:
            return "Auto exposure";
        case FlyCapture2::SHARPNESS:
            return "Sharpness";
        case FlyCapture2::WHITE_BALANCE:
            return "White balance";
        case FlyCapture2::HUE:
            return "Hue";
        case FlyCapture2::SATURATION:
            return "Saturation";
        case FlyCapture2::GAMMA:
            return "Gamma";
        case FlyCapture2::IRIS:
            return "Iris";
        case FlyCapture2::FOCUS:
            return "Focus";
        case FlyCapture2::ZOOM:
            return "Zoom";
        case FlyCapture2::PAN:
            return "Pan";
        case FlyCapture2::TILT:
            return "Tilt";
        case FlyCapture2::SHUTTER:
            return "Shutter";
        case FlyCapture2::GAIN:
            return "Gain";
        case FlyCapture2::TRIGGER_MODE:
            return "Trigger mode";
        case FlyCapture2::TRIGGER_DELAY:
            return "Trigger delay";
        case FlyCapture2::FRAME_RATE:
            return "Frame rate";
        case FlyCapture2::TEMPERATURE:
            return "Temperature";
        case FlyCapture2::UNSPECIFIED_PROPERTY_TYPE:
            return "Unspecified property type";
        default:
            return "unknown";
        }
    }

    FlyCapture2::Camera** cameras;
    call_back_data** callback_data;

    FlyCapture2::Error error;
    unsigned int number_of_cameras;

};



#endif
