package com.mapmatching.common;

import java.text.SimpleDateFormat;
import java.util.Date;

import com.fasterxml.jackson.databind.deser.ValueInstantiator.Gettable;

// Provies methods to perform simple logging.
public class Logger {

  public static void Info(String message) {
    Log("[INFO]", message);
  }

  public static void Warning(String message) {
    Log("[WARNING]", message);
  }

  public static void Error(String message) {
    Log("[ERROR]", message);
  }

  public static void Log(String errorType, String message) {
    System.out.println(errorType + "      " + GetDateTimeString(new Date()) + "     " + message);
  }

  // Returns a date in a beautiful format.
  public static String GetDateTimeString(Date date) {
    SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
    return formatter.format(date);
  }
}
